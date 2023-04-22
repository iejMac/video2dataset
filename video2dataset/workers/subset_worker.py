"""creates a subset of an existing dataset inside the sample dimension"""
import time
import json
import pyarrow as pa
import traceback

import fsspec
import numpy as np
from typing import List, Any

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)


class SubsetWorker:
    """The loader class reads the shards, then the selected data is chosen and writen by the writer"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        thread_count,
        number_sample_per_shard,
        oom_shard_count,
        encode_formats,
        video_size,
        resize_mode,
        video_fps,
        audio_rate,
        captions_are_subtitles,
        detect_cuts,
        cut_detection_mode,
        cuts_are_clips,
        cut_framerates,
        cut_detector_threshold,
        cut_detector_min_scene_len,
        min_clip_length,
        max_clip_length,
        max_clip_length_strategy,
        precise_clipping,
        oom_clip_count=5,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = True
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.captions_are_subtitles = captions_are_subtitles

        self.encode_formats = encode_formats

        self.clipping_subsampler = ClippingSubsampler(
            oom_clip_count,
            encode_formats,
            min_length=min_clip_length,
            max_length=max_clip_length,
            max_length_strategy=max_clip_length_strategy,
            precise=precise_clipping,
        )
        self.cut_detection_mode = cut_detection_mode
        self.cut_framerates = cut_framerates
        self.detect_cuts = detect_cuts
        self.cut_detector_threshold = cut_detector_threshold
        self.cut_detector_min_scene_len = cut_detector_min_scene_len
        if detect_cuts:
            self.cut_detector = CutDetectionSubsampler(
                cut_detection_mode=cut_detection_mode,
                framerates=cut_framerates,
                threshold=cut_detector_threshold,
                min_scene_len=cut_detector_min_scene_len,
            )
        self.cuts_are_clips = cuts_are_clips
        self.noop_subsampler = NoOpSubsampler()
        self.cut_detector_downsampler = ResolutionSubsampler(video_size=64, resize_mode="scale")

        video_subsamplers: List[Any] = []
        if resize_mode is not None:
            video_subsamplers.append(ResolutionSubsampler(video_size, resize_mode))
        if video_fps > 0:
            video_subsamplers.append(FrameSubsampler(video_fps))

        audio_subsamplers: List[Any] = []
        if audio_rate > 0:
            audio_subsamplers.append(AudioRateSubsampler(audio_rate, encode_formats))

        self.subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    def __call__(
        self,
        row,
    ):
        try:
            self.process_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def process_shard(
        self,
        row,
    ):
        """Function to start an video processing in one process"""

        shard, shard_id = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")
        with fs.open(shard_path, "rb") as f:
            df = pa.parquet.read_table(f)
            schema = df.schema

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.oom_shard_count,
            schema,
            self.encode_formats,
        )

        successes = 0
        failed_to_subsample = 0
        error_message = None

        if shard.startswith("s3://"):
            shard = f"pipe:aws s3 cp {shard} -"

        dataloader = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
        )
        count = 0
        for sample in dataloader:
            count += 1
            key = sample["__key__"]
            caption = sample.get("txt", b"").decode("utf-8")
            meta = json.loads(sample.get("json", b"{}").decode("utf-8"))
            streams = {}
            for mod, fmt in self.encode_formats.items():
                streams[mod] = sample[fmt]

            if self.captions_are_subtitles:  # create clips
                subtitles = meta["yt_meta_dict"]["subtitles"]
                meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]

            elif self.detect_cuts:  # apply cut detection to get clips
                video_bytes = streams["video"]
                downsampled_video_bytes, error_message = self.cut_detector_downsampler([video_bytes])
                if error_message is not None:
                    failed_to_subsample += 1
                    status = "failed_to_subsample"
                    status_dict.increment(error_message)
                    meta["status"] = status
                    meta["error_message"] = error_message

                    sample_writer.write(
                        {},
                        key,
                        caption,
                        meta,
                    )
                    continue
                meta["cuts"] = self.cut_detector(downsampled_video_bytes[0])

            if self.cuts_are_clips:
                cuts = meta["cuts"]
                native_fps = cuts["original_fps"]
                meta["clips"] = (np.array(cuts["cuts_original_fps"]) / native_fps).tolist()

            # 1 video -> many videos (either clipping or noop which does identity broadcasting)
            broadcast_subsampler = (
                self.clipping_subsampler
                if (self.captions_are_subtitles or self.cuts_are_clips)
                else self.noop_subsampler
            )
            subsampled_streams, metas, error_message = broadcast_subsampler(streams, meta)
            for modality in subsampled_streams:
                for modality_subsampler in self.subsamplers[modality]:
                    subsampled_modality, error_message = modality_subsampler(subsampled_streams[modality])
                    subsampled_streams[modality] = subsampled_modality

            if error_message is not None:
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message

                sample_writer.write(
                    {},
                    key,
                    caption,
                    meta,
                )
                continue

            successes += 1
            status = "success"
            status_dict.increment(status)
            subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
            if len(subsampled_streams_list) == 0:  # no audio or video, just write meta
                meta["status"] = status
                sample_writer.write(
                    {},
                    key,
                    caption,
                    meta,
                )
                continue

            for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                meta["status"] = status

                text_caption = caption
                if self.captions_are_subtitles:
                    text_caption = meta["yt_meta_dict"].pop("subtitles")

                sample_writer.write(
                    subsampled_streams,
                    meta["key"],
                    text_caption,
                    meta,
                )
            """
            except Exception as err:  # pylint: disable=broad-except
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message

                sample_writer.write(
                    {},
                    key,
                    caption,
                    meta,
                )
                traceback.print_exc()
                print(f"Sample {key} failed to download: {err}")
            """
        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
