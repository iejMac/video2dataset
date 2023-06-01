"""Create dataset from video links and metadata."""
import os
import sys
import signal
import fire
import fsspec

from omegaconf import OmegaConf
from typing import List, Optional, Union, Dict, Any
import numpy as np  # pylint: disable=unused-import

from .logger import LoggerProcess
from .data_writer import (
    WebDatasetSampleWriter,
    FilesSampleWriter,
    ParquetSampleWriter,
    TFRecordSampleWriter,
    DummySampleWriter,
)
from .input_sharder import InputSharder
from .output_sharder import OutputSharder
from .distributor import (
    multiprocessing_distributor,
    pyspark_distributor,
    SlurmDistributor,
    SlurmShardSampler,
)
from .workers import DownloadWorker, SubsetWorker, OpticalFlowWorker
from .configs import CONFIGS


def identity(x):
    return x


# pylint: disable=unused-argument
# pylint: disable=eval-used
# pylint: disable=broad-except
def video2dataset(
    url_list: str,
    output_folder: str = "videos",
    output_format: str = "files",
    input_format: str = "txt",
    encode_formats: dict = None,
    stage: str = "download",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    save_additional_columns: Optional[List[str]] = None,
    enable_wandb: bool = False,
    wandb_project: str = "video2dataset",
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    tmp_dir: str = "/tmp",
    config: Union[str, Dict[Any]] = "default",
):
    """
    create video dataset from video links
    """
    local_args = dict(locals())
    if isinstance(config, str):
        config = CONFIGS[config] if config in CONFIGS else OmegaConf.load(config)
        config = OmegaConf.to_container(config)
    for arg_type in ["subsampling", "reading", "storage", "distribution"]:
        assert arg_type in config

    if config["reading"]["sampler"] is None:
        config["reading"]["sampler"] = identity

    called_from_slurm = "CALLED_FROM_SLURM" in os.environ
    if called_from_slurm:
        global_task_id = int(os.environ["GLOBAL_RANK"])
        num_tasks = (
            config["distribution"]["distributor_args"]["n_nodes"]
            * config["distribution"]["distributor_args"]["tasks_per_node"]
        )
        config["reading"]["sampler"] = SlurmShardSampler(global_task_id=global_task_id, num_tasks=num_tasks)

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if config["storage"]["captions_are_subtitles"]:
        assert clip_col is None  # no weird double-clipping
        if config["reading"]["yt_args"]["yt_metadata_args"] is None:
            config["reading"]["yt_args"]["yt_metadata_args"] = {}
        config["reading"]["yt_args"]["yt_metadata_args"]["writesubtitles"] = True

    if encode_formats is None:
        encode_formats = {"video": "mp4"}

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, local_args)
    tmp_path = output_folder + "/_tmp"
    fs, run_tmp_dir = fsspec.core.url_to_fs(tmp_path)
    if not fs.exists(run_tmp_dir):
        fs.mkdir(run_tmp_dir)

    def signal_handler(signal_arg, frame):  # pylint: disable=unused-argument
        try:
            fs.rm(run_tmp_dir, recursive=True)
        except Exception as _:  # pylint: disable=broad-except
            pass
        logger_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    save_caption = caption_col is not None or config["storage"]["captions_are_subtitles"]

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

    if stage == "download":
        shard_iterator = InputSharder(  # type: ignore
            url_list,
            input_format,
            url_col,
            caption_col,
            clip_col,
            save_additional_columns,
            config["storage"]["number_sample_per_shard"],
            done_shards,
            tmp_path,
            config["reading"]["sampler"],
        )
        worker = DownloadWorker(
            sample_writer_class=sample_writer_class,
            save_caption=save_caption,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            tmp_dir=tmp_dir,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "subset":
        shard_iterator = OutputSharder(
            url_list, input_format, done_shards, sampler=config["reading"]["sampler"]  # type: ignore
        )

        worker = SubsetWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            config=config,
        )
    elif stage == "optical_flow":
        shard_iterator = OutputSharder(  # type: ignore
            url_list, input_format, done_shards, sampler=config["reading"]["sampler"]
        )
        is_slurm_task = "GLOBAL_RANK" in os.environ and distributor == "multiprocessing"
        worker = OpticalFlowWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            encode_formats=encode_formats,
            is_slurm_task=is_slurm_task,
            config=config,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")

    print("Starting the downloading of this file")
    # TODO: slurm distributor sets arg distributor but that won't work here because of config
    # I Think we can just set called_from_slurm normally here and based on that go into multiproc
    # TODO: while you're at it fix the problem where each worker logs, when you set multiproc to true
    # in spawned slurm procs also sent their enable_wandb to false unless its master worker
    if config["distribution"]["distributor"] == "multiprocessing" or called_from_slurm:
        distributor_fn = multiprocessing_distributor
        called_from_slurm = "GLOBAL_RANK" in os.environ
    elif config["distribution"]["distributor"] == "pyspark":
        distributor_fn = pyspark_distributor
    elif config["distribution"]["distributor"] == "slurm":
        worker_args = {key: local_args[key] for key in local_args if not key.startswith("slurm")}
        slurm_args = config["distribution"]["distributor_args"]

        distributor_fn = SlurmDistributor(worker_args=worker_args, **slurm_args)
    else:
        raise ValueError(f"Distributor {config['distribution']['distributor']} not supported")

    distributor_fn(
        config["distribution"]["processes_count"],
        worker,
        shard_iterator,
        config["distribution"]["subjob_size"],
        max_shard_retry,
    )
    logger_process.join()
    if not called_from_slurm:
        fs.rm(run_tmp_dir, recursive=True)


def main():
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
