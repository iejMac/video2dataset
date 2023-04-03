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

def quantize_endpoints(frame_intervals, native_fps, detected_fps):
    quantized_intervals = []
    factor = native_fps // detected_fps - 2

    for start, end in frame_intervals:
        quantized_start = max(0, start + factor)
        quantized_end = end - factor
        quantized_intervals.append([quantized_start, quantized_end])

    return quantized_intervals

def interval_intersection(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    start_max = max(start1, start2)
    end_min = min(end1, end2)

    if start_max <= end_min:
        return [start_max, end_min]
    else:
        return None

def combine_two_intervals(list1, list2):
    combined_list = []

    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        intersection = interval_intersection(list1[i], list2[j])
        if intersection is not None:
            combined_list.append(intersection)

        if list1[i][1] < list2[j][1]:
            i += 1
        else:
            j += 1

    return combined_list

def combine_multiple_intervals(lists_of_intervals):
    if not lists_of_intervals:
        return []

    combined_list = lists_of_intervals[0]
    for i in range(1, len(lists_of_intervals)):
        combined_list = combine_two_intervals(combined_list, lists_of_intervals[i])

    return combined_list

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
        clipping_mode,
        cut_framerates,
        oom_clip_count=5,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.encode_formats = encode_formats
        self.save_caption = True

        self.captions_are_subtitles = captions_are_subtitles
        self.clipping_subsampler = ClippingSubsampler(oom_clip_count, encode_formats)
        self.cut_detection_mode = cut_detection_mode
        self.cut_framerates = cut_framerates
        self.detect_cuts = detect_cuts
        if detect_cuts:
            self.cut_detector = CutDetectionSubsampler(cut_detection_mode=cut_detection_mode, framerates=cut_framerates)
        self.cuts_are_clips = cuts_are_clips
        self.clipping_mode = clipping_mode
        self.noop_subsampler = NoOpSubsampler()

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
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )

        successes = 0
        failed_to_subsample = 0
        error_message = None

        if 's3' in shard:
            shard = f'pipe:aws s3 cp {shard} -'

        dataloader = get_video_dataset(
            urls=[shard],
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
        )
        for sample in dataloader:
            # Gather subset of dataset
            key = sample["__key__"]
            caption = sample.get("txt", b"").decode("utf-8")
            meta = json.loads(sample.get("json", b"{}").decode("utf-8"))
            streams = {}
            for mod, fmt in self.encode_formats.items():
                streams[mod] = sample[fmt]

            if self.captions_are_subtitles:  # create clips
                subtitles = meta["yt_meta_dict"]["subtitles"]
                meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                meta["lines"] = [" ".join(line_dict["lines"]) for line_dict in subtitles]

            elif self.detect_cuts:  # apply cut detection to get clips
                detected_cuts = self.cut_detector(streams)
                if "cuts" not in meta:
                    meta["cuts"] = detected_cuts
                else:
                    for k in detected_cuts:
                        if k not in meta["cuts"]:
                            meta["cuts"][k] = detected_cuts[k]

            if self.cuts_are_clips:
                cuts = meta["cuts"]
                native_fps = cuts["original_fps"]
                if self.clipping_mode == "default":
                    cuts = (np.array(cuts["cuts_original_fps"]) / native_fps).tolist()
                elif self.clipping_mode == "quantize":
                    quantized_cuts = []
                    for k in meta["cuts"]:
                        if "cuts" in k and k != "cuts_original_fps":
                            cut_fps = int(k.split('_')[-1])
                            quantized = quantize_endpoints(cuts[k], native_fps, cut_fps)
                            quantized_cuts.append(quantized)
                    all_intervals = quantized_cuts + [cuts["cuts_original_fps"]]
                    cuts = combine_multiple_intervals(all_intervals)
                if len(cuts) == 0:
                    cuts = [[0, 0]]
                meta["clips"] = (np.array(cuts)/native_fps).tolist()


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
            subsampled_streams_list = [
                dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())
            ]
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

        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            1,  # count
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
