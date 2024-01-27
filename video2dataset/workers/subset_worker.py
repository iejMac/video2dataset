"""creates a subset of an existing dataset inside the sample dimension"""
import fsspec
import json
import pyarrow as pa
import time
import traceback
from typing import Literal, cast
import webdataset as wds

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import write_stats
from video2dataset.types import EncodeFormats, Streams
from video2dataset.workers.worker import ShardStatus, get_subsamplers, process_sample


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
        self.input_encode_formats = encode_formats
        self.subsamplers, self.output_encode_formats = get_subsamplers(config, self.input_encode_formats)

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
            print(f"shard_file {row[0]} failed with error {err}")
            return (False, row)

    def get_shard_processors(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Get objects for loading and writing data"""

        try:
            fs, shard_path = fsspec.core.url_to_fs(shard_file[: -len(".tar")] + ".parquet")
            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception:  # pylint: disable=broad-except
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
            urls=shard_file,
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
            handler=wds.warn_and_continue,
        )
        return shard_sample_writer, shard_dataloader

    def process_shard(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Function to start an video processing in one process"""

        start_time = time.time()
        shard_sample_writer, shard_dataloader = self.get_shard_processors(shard_file, shard_id)
        shard_status = ShardStatus()

        for sample in shard_dataloader:
            shard_status.count += 1
            key = sample["__key__"]
            try:
                caption = sample.get("txt", b"").decode("utf-8")
                metadata = json.loads(sample.get("json", b"{}").decode("utf-8"))
            except Exception as err:  # pylint: disable=broad-except
                traceback.print_exc()
                print(f"Sample {key} failed to download: {err}")
                return

            streams: Streams = {}
            for modality, encode_format in self.input_encode_formats.items():
                modality = cast(Literal["audio", "video"], modality)
                streams[modality] = [sample[encode_format]]

            process_sample(
                subsamplers=self.subsamplers,
                shard_status=shard_status,
                streams=streams,
                key=key,
                caption=caption,
                metadata=metadata,
                captions_are_subtitles=self.config["storage"]["captions_are_subtitles"],
                shard_sample_writer=shard_sample_writer,
            )

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
