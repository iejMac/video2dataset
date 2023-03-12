"""the downloader module handles the downloading"""

import math
import time
import pyarrow as pa
import traceback

import fsspec

from multiprocessing.pool import ThreadPool
from threading import Semaphore
from typing import List, Any

from video2dataset.dataloader import get_bytes_dataloader
from .logger import CappedCounter
from .logger import write_stats
from .subsamplers import ClippingSubsampler, FrameSubsampler, NoOpSubsampler, ResolutionSubsampler, AudioRateSubsampler


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class DummyWorker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        thread_count,
        number_sample_per_shard,
        oom_shard_count,
        tmp_dir,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count

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

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))

        with ThreadPool(self.thread_count) as thread_pool:
            for key, streams, yt_meta_dict, error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                loader,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                        "yt_meta_dict": yt_meta_dict,
                    }

                    if error_message is not None:
                        if "[youtube]" in error_message:  # video-specific error, remove videoID
                            error_message = "ERROR: [youtube]:" + error_message.split(":")[-1]
                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue

                    for stream in streams.values():
                        bytes_downloaded += len(stream)

                    metas = [meta]

                    if self.captions_are_subtitles:  # create clips
                        subtitles = meta["yt_meta_dict"]["subtitles"]
                        meta["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                        meta["lines"] = [" ".join(line_dict["lines"]) for line_dict in subtitles]

                    # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                    broadcast_subsampler = (
                        self.clipping_subsampler
                        if ("clips" in self.column_list or self.captions_are_subtitles)
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
                        meta["clips"] = []
                        meta["error_message"] = error_message
                        sample_writer.write(
                            {},
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        semaphore.release()
                        continue

                    successes += 1
                    status = "success"
                    status_dict.increment(status)
                    subsampled_streams_list = [
                        dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())
                    ]

                    for subsampled_streams, meta in zip(subsampled_streams_list, metas):
                        meta["status"] = status

                        text_caption = (sample_data[caption_indice] if caption_indice is not None else None,)
                        if self.captions_are_subtitles:
                            text_caption = meta["yt_meta_dict"].pop("subtitles")

                        sample_writer.write(
                            subsampled_streams,
                            meta["key"],
                            text_caption,
                            meta,
                        )
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed_to_download,
            failed_to_subsample,
            bytes_downloaded,
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
        fs.rm(shard_path)
