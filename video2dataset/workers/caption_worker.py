"""
Worker for caption stage
"""
import time
import pyarrow as pa
import traceback
import io
import numpy as np
import fsspec
import webdataset as wds

from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import CaptionSubsampler
from video2dataset.dataloader import get_video_dataset


class CaptionWorker:
    """
    A class to read shards, process them using CaptionSubsampler, and write the output.

    Attributes:
        sample_writer_class (type): The class used to write samples.
        output_folder (str): The folder to write the output.
        thread_count (int): The number of threads.
        number_sample_per_shard (int): The number of samples per shard.
        oom_shard_count (int): The number of out-of-memory shards.
        encode_formats (dict): The encoding formats.
        fps (int): The target frames per second.
        caption_subsampler (CaptionSubsampler): The CaptionSubsampler instance.
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

        self.caption_subsampler = CaptionSubsampler(
            **self.config["subsampling"]["CaptionSubsampler"]["args"],
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
        Process a video shard using the CaptionSubsampler.

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
            decoder_kwargs=self.config["reading"]["dataloader_args"]["decoder_kwargs"],
            resize_size=self.config["reading"]["dataloader_args"]["resize_size"],
            crop_size=None,
            enforce_additional_keys=[],
            return_always=True,
            handler=wds.warn_and_continue,
        )
        count = 0
        for sample in dset:
            count += 1
            corrupted = sample["__corrupted__"][0]
            key = sample["__key__"][0]
            dummy_cap = ""
            if corrupted:
                url = sample["__url__"][0]
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
                    {"caption": dummy_cap},
                    key,
                    None,
                    meta,
                )
                continue
            meta = sample["json"][0]
            streams = {}
            frames = np.array(sample.get("mp4")[0]).astype(np.float32)
            caption, error_message = self.caption_subsampler(frames)

            if error_message is not None:
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message
                meta["__corrupted__"] = True
                sample_writer.write(
                    {"caption": dummy_cap},
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

            streams["caption"] = caption
            sample_writer.write(
                streams,
                key,
                None,
                meta,
            )

        sample_writer.close()
        end_time = time.time()

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
