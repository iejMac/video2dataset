"""creates a subset of an existing dataset inside the sample dimension"""
from dataclasses import dataclass, field
import time
import json
import pyarrow as pa
import traceback

import fsspec
import numpy as np
import webdataset as wds
from typing import List, Any, Union, Optional

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)
from video2dataset.types import EncodeFormats, Streams


@dataclass
class Subsamplers:
    broadcast_subsampler: Union[ClippingSubsampler, NoOpSubsampler]


def get_subsamplers(config: dict, encode_formats: EncodeFormats):
    clipping_subsampler = ClippingSubsampler(
        5,  # oom_clip_count
        encode_formats,
        **config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
    )

    need_keyframes = clipping_subsampler.precision == "keyframe_adjusted"
    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(
            **config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"]
        )
        ffprobe_subsampler.extract_keyframes |= need_keyframes
    noop_subsampler = NoOpSubsampler()
    video_subsamplers: List[Any] = []
    if "ResolutionSubsampler" in config["subsampling"]:
        video_subsamplers.append(ResolutionSubsampler(**config["subsampling"]["ResolutionSubsampler"]["args"]))
    if "FrameSubsampler" in config["subsampling"]:
        video_subsamplers.append(FrameSubsampler(**config["subsampling"]["FrameSubsampler"]["args"]))

    audio_subsamplers: List[Any] = []
    if "AudioRateSubsampler" in config["subsampling"]:
        audio_subsamplers.append(AudioRateSubsampler(**config["subsampling"]["AudioRateSubsampler"]["args"]))
    subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    cut_detection_subsampler = None
    cuts_are_clips = False
    if "CutDetectionSubsampler" in config["subsampling"]:
        if "args" in config["subsampling"]["CutDetectionSubsampler"]:
            cut_detection_subsampler = CutDetectionSubsampler(
                **config["subsampling"]["CutDetectionSubsampler"]["args"]
            )
        cuts_are_clips = config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

    broadcast_subsampler = (
        clipping_subsampler
        if (config["storage"]["captions_are_subtitles"] or cuts_are_clips)
        else noop_subsampler
    )

    return ffprobe_subsampler, subsamplers, cut_detection_subsampler, cuts_are_clips, broadcast_subsampler


@dataclass
class ShardStatus:
    successes: int = 0
    failed: dict = field(
        default_factory=lambda: {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
    )
    status_dict: CappedCounter = field(
        default_factory=CappedCounter
    )
    error_message: Optional[str] = None
    count: int = 0


class SubsetWorker:
    """The loader class reads the shards, then the selected data is chosen and writen by the writer"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        encode_formats: EncodeFormats,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.config = config
        self.ffprobe_subsampler, self.subsamplers, self.cut_detection_subsampler, self.cuts_are_clips, self.broadcast_subsampler = get_subsamplers(config, encode_formats)

        # set encoding formats
        self.input_encode_formats = encode_formats
        self.output_encode_formats = self.input_encode_formats.copy()
        if self.subsamplers["audio"]:
            assert (
                len({s.encode_format for s in self.subsamplers["audio"]}) == 1
            )  # assert that all audio subsamplers have the same output format
            self.output_encode_formats["audio"] = self.subsamplers["audio"][0].encode_format
        if self.subsamplers["video"]:
            assert (
                len({s.encode_format for s in self.subsamplers["video"]}) == 1
            )  # assert that all video subsamplers have the same output format
            self.output_encode_formats["video"] = self.subsamplers["video"][0].encode_format


    def __call__(
        self,
        row,
    ):
        try:
            shard, shard_id = row
            self.process_shard(shard, shard_id)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def get_shard_processors(self, shard, shard_id):
        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")

            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception as e:  # pylint: disable=broad-except,unused-variable
            fields = [
                pa.field("key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("error_message", pa.string()),
            ]
            schema = pa.schema(fields)
        shard_sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            True,  # save_caption
            self.config["storage"]["oom_shard_count"],
            schema,
            self.output_encode_formats,
        )
        shard_dataloader = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
            handler=wds.warn_and_continue,
        )
        return shard_sample_writer, shard_dataloader

    def process_shard(
        self,
        shard,
        shard_id,
    ):
        """Function to start an video processing in one process"""

        start_time = time.time()
        shard_sample_writer, shard_dataloader = self.get_shard_processors(shard, shard_id)
        shard_status = ShardStatus()

        for sample in shard_dataloader:
            try:
                shard_status.count += 1
                key = sample["__key__"]
                caption = sample.get("txt", b"").decode("utf-8")
                meta = json.loads(sample.get("json", b"{}").decode("utf-8"))
                streams = {}
                for mod, fmt in self.input_encode_formats.items():
                    streams[mod] = [sample[fmt]]

                if self.ffprobe_subsampler is not None:
                    streams, meta, shard_status.error_message = self.ffprobe_subsampler(streams, meta)
                    if shard_status.error_message is not None:
                        raise ValueError("failed_to_subsample")

                if self.config["storage"]["captions_are_subtitles"]:  # create clips
                    subtitles = meta["yt_meta_dict"]["subtitles"]
                    meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                elif self.cut_detection_subsampler is not None:  # apply cut detection to get clips
                    streams, cuts, shard_status.error_message = self.cut_detection_subsampler(streams)
                    if shard_status.error_message is not None:
                        raise ValueError("failed_to_subsample")

                    meta["cuts"] = cuts

                if self.cuts_are_clips:
                    cuts = meta["cuts"]
                    native_fps = cuts["original_fps"]
                    meta["clips"] = (np.array(cuts["cuts_original_fps"]) / native_fps).tolist()

                # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                subsampled_streams, metas, shard_status.error_message = self.broadcast_subsampler(streams, meta)
                if shard_status.error_message is not None:
                    meta["clips"] = []
                    raise ValueError("failed_to_subsample")

                for modality in list(subsampled_streams.keys()):
                    for modality_subsampler in self.subsamplers[modality]:
                        subsampled_streams, metas, shard_status.error_message = modality_subsampler(subsampled_streams, metas)

                if shard_status.error_message is not None:
                    raise ValueError("failed_to_subsample")

                shard_status.successes += 1
                status = "success"
                shard_status.status_dict.increment(status)
                subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
                if len(subsampled_streams_list) == 0:  # no audio or video, just write meta
                    meta["status"] = status
                    shard_sample_writer.write(
                        {},
                        key,
                        caption,
                        meta,
                    )
                    continue

                for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                    meta["status"] = status

                    text_caption = caption
                    if self.config["storage"]["captions_are_subtitles"]:
                        text_caption = meta.get("clip_subtitles")[0]["lines"][0]

                    shard_sample_writer.write(
                        subsampled_streams,
                        meta["key"],
                        text_caption,
                        meta,
                    )

            except Exception as err:  # pylint: disable=broad-except
                status = str(err)
                if status.startswith("failed_to_"):
                    shard_status.failed[status] += 1
                    shard_status.status_dict.increment(shard_status.error_message)
                    meta["status"] = status
                    meta["error_message"] = shard_status.error_message
                    shard_sample_writer.write(
                        {},
                        key,
                        caption,
                        meta,
                    )
                else:
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")

        shard_sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            shard_status.count,
            shard_status.successes,
            0,  # failed to download
            shard_status.failed["failed_to_subsample"],
            0,  # bytes downloaded
            start_time,
            end_time,
            shard_status.status_dict,
            self.config["storage"]["oom_shard_count"],
        )
