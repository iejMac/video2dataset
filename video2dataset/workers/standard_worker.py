"""Standard worker for video2dataset."""
from dataclasses import dataclass, field
import fsspec
import math
from multiprocessing.pool import ThreadPool
import numpy as np
import pyarrow as pa
from threading import Semaphore
import time
import traceback
from typing import Any, List, Tuple, Optional, Type, cast

from video2dataset.data_reader import VideoDataReader
from video2dataset.data_writer import SampleWriter
from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
    Subsampler,
)
from video2dataset.types import EncodeFormats, Streams, Metadata


@dataclass
class ShardStatus:
    """Shard processing status"""

    successes: int = 0
    failed: dict = field(
        default_factory=lambda: {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
    )
    status_dict: CappedCounter = field(default_factory=CappedCounter)
    error_message: Optional[str] = None
    count: int = 0
    bytes_downloaded: int = 0


@dataclass
class Subsamplers:
    """Subsamplers used in processing"""

    ffprobe_subsampler: Optional[FFProbeSubsampler] = None
    modal_subsamplers: dict = field(default_factory=dict)
    cut_detection_subsampler: Optional[CutDetectionSubsampler] = None
    cuts_are_clips: bool = False
    broadcast_subsampler: Subsampler = field(default_factory=NoOpSubsampler)


def get_subsamplers(
    config: dict,
    input_encode_formats: EncodeFormats,
    do_clipping: bool = False,
) -> Tuple[Subsamplers, EncodeFormats]:
    """Initialize all subsamplers using config"""

    clipping_subsampler = ClippingSubsampler(
        oom_clip_count=5,
        encode_formats=input_encode_formats,
        **config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
    )
    need_keyframes = clipping_subsampler.precision == "keyframe_adjusted"

    cut_detection_subsampler = None
    cuts_are_clips = False
    if "CutDetectionSubsampler" in config["subsampling"]:
        if "args" in config["subsampling"]["CutDetectionSubsampler"]:
            cut_detection_subsampler = CutDetectionSubsampler(**config["subsampling"]["CutDetectionSubsampler"]["args"])
        cuts_are_clips = config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

    broadcast_subsampler = (
        clipping_subsampler
        if (do_clipping or config["storage"]["captions_are_subtitles"] or cuts_are_clips)
        else NoOpSubsampler()
    )

    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(**config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"])
        ffprobe_subsampler.extract_keyframes |= need_keyframes

    video_subsamplers: List[Any] = []
    if "ResolutionSubsampler" in config["subsampling"]:
        video_subsamplers.append(ResolutionSubsampler(**config["subsampling"]["ResolutionSubsampler"]["args"]))
    if "FrameSubsampler" in config["subsampling"]:
        video_subsamplers.append(FrameSubsampler(**config["subsampling"]["FrameSubsampler"]["args"]))

    audio_subsamplers: List[Any] = []
    if "AudioRateSubsampler" in config["subsampling"]:
        audio_subsamplers.append(AudioRateSubsampler(**config["subsampling"]["AudioRateSubsampler"]["args"]))

    modal_subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    # output encoding formats
    output_encode_formats = input_encode_formats.copy()
    if modal_subsamplers["audio"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["audio"]}) == 1
        )  # assert that all audio subsamplers have the same output format
        output_encode_formats["audio"] = modal_subsamplers["audio"][0].encode_format
    if modal_subsamplers["video"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["video"]}) == 1
        )  # assert that all video subsamplers have the same output format
        output_encode_formats["video"] = modal_subsamplers["video"][0].encode_format

    return (
        Subsamplers(
            ffprobe_subsampler=ffprobe_subsampler,
            modal_subsamplers=modal_subsamplers,
            cut_detection_subsampler=cut_detection_subsampler,
            cuts_are_clips=cuts_are_clips,
            broadcast_subsampler=broadcast_subsampler,
        ),
        output_encode_formats,
    )


def process_sample(
    subsamplers: Subsamplers,
    shard_status: ShardStatus,
    streams: Streams,
    key: str,
    caption: str,
    metadata: Metadata,
    captions_are_subtitles: bool,
    shard_sample_writer: Any,  # TODO: type correctly
):
    try:
        if subsamplers.ffprobe_subsampler is not None:
            streams, metadata, shard_status.error_message = subsamplers.ffprobe_subsampler(streams, metadata)
            assert shard_status.error_message is None

        if captions_are_subtitles:  # create clips
            subtitles = metadata["yt_meta_dict"]["subtitles"]
            metadata["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
        elif subsamplers.cut_detection_subsampler is not None:  # apply cut detection to get clips
            streams, cuts, shard_status.error_message = subsamplers.cut_detection_subsampler(streams)
            assert shard_status.error_message is None
            metadata["cuts"] = cuts
            assert cuts is not None
            if subsamplers.cuts_are_clips:
                metadata["clips"] = (np.array(cuts["cuts_original_fps"]) / cuts["original_fps"]).tolist()

        # 1 video -> many videos (either clipping or noop which does identity broadcasting)
        subsampled_streams, metadatas, shard_status.error_message = subsamplers.broadcast_subsampler(streams, metadata)
        if shard_status.error_message is not None:
            metadata["clips"] = []
            assert False

        for modality in list(subsampled_streams.keys()):
            for modality_subsampler in subsamplers.modal_subsamplers[modality]:
                subsampled_streams, metadatas, shard_status.error_message = modality_subsampler(
                    subsampled_streams, metadatas
                )
                assert shard_status.error_message is None

        shard_status.successes += 1
        status = "success"
        shard_status.status_dict.increment(status)

        subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
        if len(subsampled_streams_list) == 0:  # no audio or video, just write metadata
            metadata["status"] = status
            shard_sample_writer.write(
                {},
                key,
                caption,
                metadata,
            )
            return
        for subsampled_streams, subsampled_metadata in zip(subsampled_streams_list, metadatas):
            subsampled_metadata["status"] = status
            text_caption = caption
            if captions_are_subtitles:
                clip_subtitles = subsampled_metadata.get("clip_subtitles")
                first_clip_subtitles = clip_subtitles[0] if clip_subtitles else None
                subtitle_lines = first_clip_subtitles["lines"] if first_clip_subtitles else None
                text_caption = subtitle_lines[0] if subtitle_lines else text_caption
            shard_sample_writer.write(
                subsampled_streams,
                subsampled_metadata["key"],
                text_caption,
                subsampled_metadata,
            )
    except Exception as err:  # pylint: disable=broad-except
        print(err)
        shard_status.failed["failed_to_subsample"] += 1
        shard_status.status_dict.increment(shard_status.error_message)
        metadata["status"] = "failed_to_subsample"
        metadata["error_message"] = shard_status.error_message
        shard_sample_writer.write(
            {},
            key,
            caption,
            metadata,
        )


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class StandardWorker:
    """Download and process shards with threads"""

    def __init__(
        self,
        sample_writer_class: Type[SampleWriter],
        output_folder: str,
        encode_formats: EncodeFormats,
        config: dict,
        # for downloading
        save_caption: bool,
        column_list: Optional[List[str]],
        tmp_dir: str,
    ) -> None:
        # the following options are used for downloading
        self.save_caption = save_caption
        self.column_list = column_list if column_list is not None else ["url"]
        self.url_indice = self.column_list.index("url")
        self.caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        self.data_reader = VideoDataReader(encode_formats, tmp_dir, config["reading"])
        self.oom_sample_per_shard = math.ceil(math.log10(self.config["storage"]["number_sample_per_shard"]))
        # the following options are used for processing
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.config = config
        self.input_encode_formats = encode_formats
        self.subsamplers, self.output_encode_formats = get_subsamplers(
            config,
            self.input_encode_formats,
            do_clipping=("clips" in self.column_list),
        )

    def __call__(
        self,
        row,
    ):
        try:
            shard_file, shard_id = row
            self.process_shard(shard_file, shard_id)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def get_shard_processors(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Get objects for loading and writing data"""

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        print(shard_path)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
            schema = df.schema
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
        )
        shard_sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            self.output_encode_formats,
        )
        pydict = df.select(self.column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))

        def rm_shard_path():
            fs.rm(shard_path)

        return shard_sample_writer, shard_to_dl, rm_shard_path

    def process_shard(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Function to start an video downloading in one process"""

        start_time = time.time()
        shard_sample_writer, shard_to_dl, rm_shard_path = self.get_shard_processors(shard_file, shard_id)
        shard_status = ShardStatus(count=len(shard_to_dl))

        semaphore = Semaphore(self.config["distribution"]["thread_count"])

        def data_generator():
            for key_and_url in [(key, x[self.url_indice]) for key, x in shard_to_dl]:
                with semaphore:
                    yield key_and_url

        data_reader_call_param_generator = data_generator()

        with ThreadPool(self.config["distribution"]["thread_count"]) as thread_pool:
            for key, streams, yt_meta_dict, shard_status.error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                data_reader_call_param_generator,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(
                        key, shard_id, self.oom_sample_per_shard, self.config["storage"]["oom_shard_count"]
                    )
                    caption = sample_data[self.caption_indice] if self.caption_indice is not None else None
                    metadata = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": shard_status.error_message,
                        "yt_meta_dict": yt_meta_dict,
                    }
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                    return

                try:
                    if shard_status.error_message is not None:
                        print(shard_status.error_message)
                        if "[youtube]" in shard_status.error_message:  # video-specific error, remove videoID
                            shard_status.error_message = "ERROR: [youtube]:" + shard_status.error_message.split(":")[-1]
                        raise ValueError
                except Exception:  # pylint: disable=broad-except
                    shard_status.failed["failed_to_download"] += 1
                    shard_status.status_dict.increment(shard_status.error_message)
                    metadata["status"] = "failed_to_download"
                    metadata["error_message"] = shard_status.error_message
                    shard_sample_writer.write(
                        {},
                        str_key,
                        sample_data[self.caption_indice] if self.caption_indice is not None else None,
                        metadata,
                    )
                    return

                for stream in streams.values():
                    shard_status.bytes_downloaded += len(stream)
                for modality in streams:
                    streams[modality] = [streams[modality]]

                process_sample(
                    subsamplers=self.subsamplers,
                    shard_status=shard_status,
                    streams=cast(Streams, streams),
                    key=str_key,
                    caption=cast(str, caption),
                    metadata=metadata,
                    captions_are_subtitles=self.config["storage"]["captions_are_subtitles"],
                    shard_sample_writer=shard_sample_writer,
                )

            shard_sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool
        rm_shard_path()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            shard_status.count,
            shard_status.successes,
            shard_status.failed["failed_to_download"],
            shard_status.failed["failed_to_subsample"],
            shard_status.bytes_downloaded,
            start_time,
            end_time,
            shard_status.status_dict,
            self.config["storage"]["oom_shard_count"],
        )
