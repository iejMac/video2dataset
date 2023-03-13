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
from video2dataset.logger import CappedCounter
from video2dataset.logger import write_stats
from video2dataset.subsamplers import ClippingSubsampler, FrameSubsampler, NoOpSubsampler, ResolutionSubsampler, AudioRateSubsampler


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

        # TODO: clean this up
        self.save_caption=True
        self.encode_formats = {"video": "mp4", "audio": "mp3"}

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

        fs, shard_path = fsspec.core.url_to_fs(shard[:-len(".tar")] + ".parquet")
        with fs.open(shard_path, "rb") as f:
            df = pa.parquet.read_table(f)
            schema = df.schema

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))

        successes = 0 
        failed_to_subsample = 0
        error_message = None

        dataloader = get_bytes_dataloader([shard])
        for key, video, text, meta in dataloader:

            # Do your subsampling:
            video = video

            if error_message is not None:
                failed_to_transform += 1
                status = "failed_to_subsample"
                continue
            successes += 1
            status = "success"
            status_dict.increment(status)

        end_time = time.time()
