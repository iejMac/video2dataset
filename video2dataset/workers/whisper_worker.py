"""
Worker for optical flow stage
"""
import time
import pyarrow as pa
import traceback
import io
import json
import numpy as np
import fsspec
import webdataset as wds

from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import WhisperSubsampler
from video2dataset.dataloader import get_video_dataset


class WhisperWorker:
    """
    A class to read shards, process them using WhisperSubsampler, and write the output.

    Attributes:
        sample_writer_class (type): The class used to write samples.
        output_folder (str): The folder to write the output.
        thread_count (int): The number of threads.
        number_sample_per_shard (int): The number of samples per shard.
        oom_shard_count (int): The number of out-of-memory shards.
        encode_formats (dict): The encoding formats.
        detector (str): The optical flow detector type.
        fps (int): The target frames per second.
        optical_flow_subsampler (WhisperSubsampler): The WhisperSubsampler instance.
    """

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        encode_formats,
        is_slurm_task,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.encode_formats = encode_formats
        self.save_caption = False
        self.config = config

        self.whisper_subsampler = WhisperSubsampler(
            **self.config["subsampling"]["WhisperSubsampler"]["args"],
            is_slurm_task=is_slurm_task,
        )

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
        """
        Process a video shard using the WhisperSubsampler.

        Args:
            row (tuple): A tuple containing the shard and shard_id.

        Raises:
            Except
        """
        shard, shard_id = row
        start_time = time.time()

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

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            self.encode_formats,
        )

        successes = 0
        failed_to_subsample = 0

        dset = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs={},
            video_key=self.encode_formats["audio"],
            enforce_additional_keys=[],
            return_always=True,
            handler=wds.warn_and_continue,
        )
        count = 0
        for sample in dset:
            count += 1
            corrupted = sample["__corrupted__"]
            key = sample["__key__"]
            if corrupted:
                url = sample["__url__"]
                meta = {}
                meta["url"] = url
                meta["key"] = key
                error_message = "corrupted sample"
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message
                meta["__corrupted__"] = True
                sample_writer.write(
                    {},
                    key,
                    None,
                    meta,
                )
                continue
            meta = [json.loads(sample.get("json", b"{}").decode("utf-8"))]
            streams = {"audio": [sample[self.encode_formats["audio"]]]}
            streams, meta, error_message = self.whisper_subsampler(streams, meta)

            streams.pop("audio")  # only write metadata shards
            meta = meta[0]  # remove when unifying workers (needs to be list)

            if error_message is not None:
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message
                meta["__corrupted__"] = True
                sample_writer.write(
                    {},
                    key,
                    None,
                    meta,
                )
                continue

            successes += 1
            status = "success"
            status_dict.increment(status)
            meta["status"] = status
            meta["__corrupted__"] = False

            sample_writer.write(
                streams,
                key,
                None,
                meta,
            )
        sample_writer.close()
        end_time = time.time()
        return

        write_stats(
            self.output_folder,
            shard_id,
            count,  # count
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.config["storage"]["oom_shard_count"],
        )
