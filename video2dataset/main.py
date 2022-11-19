"""Create dataset from video links and metadata."""


from typing import List, Optional
import fire

from .logger import LoggerProcess
from .data_writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from .input_sharder import InputSharder
from .distributor import multiprocessing_distributor, pyspark_distributor
import fsspec
import sys
import signal
import os
from .worker import Worker


def video2dataset(
    url_list: str,
    output_folder: str = "videos",
    processes_count: int = 1,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    number_sample_per_shard: int = 10000,
    save_additional_columns: Optional[List[str]] = None,
    enable_wandb: bool = False,
    wandb_project: str = "video2dataset",
    oom_shard_count: int = 5,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    retries: int = 0,
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    timeout: int = 60,
):
    """
    create video dataset from video links
    """

    config_parameters = dict(locals())

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters)

    tmp_path = output_folder + "/_tmp"
    fs, tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(tmp_dir):
        fs.mkdir(tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None

    fs, output_path = fsspec.core.url_to_fs(output_folder)

    if not fs.exists(output_path):
        fs.mkdir(output_path)
        done_shards = set()
    else:
        if incremental_mode == "incremental":
            done_shards = set(int(x.split("/")[-1].split("_")[0]) for x in fs.glob(output_path + "/*.json"))
        elif incremental_mode == "overwrite":
            fs.rm(output_path, recursive=True)
            fs.mkdir(output_path)
            done_shards = set()
        else:
            raise ValueError(f"Unknown incremental mode {incremental_mode}")

    logger_process.done_shards = done_shards
    logger_process.start()

    input_sharder = InputSharder(
        url_list,
        input_format,
        url_col,
        caption_col,
        save_additional_columns,
        number_sample_per_shard,
        done_shards,
        tmp_path,
    )

    if output_format == "webdataset":
        sample_writer_class = WebDatasetSampleWriter
    elif output_format == "parquet":
        sample_writer_class = ParquetSampleWriter  # type: ignore
    elif output_format == "files":
        sample_writer_class = FilesSampleWriter  # type: ignore
    elif output_format == "tfrecord":
        sample_writer_class = TFRecordSampleWriter  # type: ignore
    elif output_format == "dummy":
        sample_writer_class = DummySampleWriter  # type: ignore
    else:
        raise ValueError(f"Invalid output format {output_format}")

    worker = Worker(
        sample_writer_class=sample_writer_class,
        save_caption=save_caption,
        output_folder=output_folder,
        column_list=input_sharder.column_list,
        timeout=timeout,
        number_sample_per_shard=number_sample_per_shard,
        oom_shard_count=oom_shard_count,
        encode_format="mp4",
        retries=retries,
    )

    print("Starting the downloading of this file")
    if distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
    elif distributor == "pyspark":
        distributor_fn = pyspark_distributor
    else:
        raise ValueError(f"Distributor {distributor} not supported")

    distributor_fn(
        processes_count,
        worker,
        input_sharder,
        subjob_size,
        max_shard_retry,
    )
    logger_process.join()
    fs.rm(tmp_dir, recursive=True)


def main():
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
