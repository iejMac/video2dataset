"""Create dataset from video links and metadata."""
import os
import sys
import signal
import fire
import fsspec

from typing import List, Optional
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
from .distributor import multiprocessing_distributor, pyspark_distributor, SlurmDistributor
from .workers import DownloadWorker, SubsetWorker, OpticalFlowWorker


def identity(x):
    return x


# pylint: disable=unused-argument
# pylint: disable=eval-used
# pylint: disable=broad-except
def video2dataset(
    url_list: str,
    output_folder: str = "videos",
    processes_count: int = 1,
    thread_count: int = 16,
    output_format: str = "files",
    input_format: str = "txt",
    url_col: str = "url",
    caption_col: Optional[str] = None,
    clip_col: Optional[str] = None,
    save_additional_columns: Optional[List[str]] = None,
    number_sample_per_shard: int = 10000,
    enable_wandb: bool = False,
    wandb_project: str = "video2dataset",
    oom_shard_count: int = 5,
    distributor: str = "multiprocessing",
    subjob_size: int = 1000,
    incremental_mode: str = "incremental",
    max_shard_retry: int = 1,
    video_size: int = 360,
    video_fps: int = -1,
    resize_mode: Optional[List[str]] = None,
    audio_rate: int = -1,
    timeout: int = 60,
    tmp_dir: str = "/tmp",
    yt_metadata_args: dict = None,
    captions_are_subtitles: bool = False,
    detect_cuts: bool = False,
    cut_detection_mode: str = "longest",
    cut_framerates: list = None,
    cuts_are_clips: bool = False,
    encode_formats: dict = None,
    stage: str = "download",
    optical_flow_params: dict = None,
    sampler=None,
    slurm_cpus_per_task: int = 1,
    slurm_job_name: str = "video2dataset",
    slurm_partition: str = None,
    slurm_n_nodes: int = 1,
    slurm_gpus_per_node: int = 8,
    slurm_account: str = None,
    slurm_tasks_per_node: int = 1,
    slurm_nodelist: str = None,
    slurm_exclude: str = None,
    slurm_cache_path: str = None,
    slurm_timeout: int = None,
    slurm_verbose_wait: bool = False,
):
    """
    create video dataset from video links
    """
    local_args = dict(locals())

    if sampler is None:
        sampler = identity

    # TODO: find better location for this code
    # TODO: figure out minimum yt_meta_args for subtitles to be added to metadata
    if captions_are_subtitles:
        assert clip_col is None  # no weird double-clipping
        if yt_metadata_args is None:
            yt_metadata_args = {}
        yt_metadata_args["writesubtitles"] = True

    config_parameters = dict(locals())

    if encode_formats is None:
        encode_formats = {"video": "mp4"}

    def make_path_absolute(path):
        fs, p = fsspec.core.url_to_fs(path)
        if fs.protocol == "file":
            return os.path.abspath(p)
        return path

    output_folder = make_path_absolute(output_folder)
    url_list = make_path_absolute(url_list)

    logger_process = LoggerProcess(output_folder, enable_wandb, wandb_project, config_parameters)
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

    save_caption = caption_col is not None or captions_are_subtitles

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
        shard_iterator = InputSharder(
            url_list,
            input_format,
            url_col,
            caption_col,
            clip_col,
            save_additional_columns,
            number_sample_per_shard,
            done_shards,
            tmp_path,
            sampler,
        )
        worker = DownloadWorker(
            sample_writer_class=sample_writer_class,
            save_caption=save_caption,
            output_folder=output_folder,
            column_list=shard_iterator.column_list,
            thread_count=thread_count,
            timeout=timeout,
            number_sample_per_shard=number_sample_per_shard,
            oom_shard_count=oom_shard_count,
            video_size=video_size,
            resize_mode=resize_mode,
            video_fps=video_fps,
            audio_rate=audio_rate,
            tmp_dir=tmp_dir,
            yt_metadata_args=yt_metadata_args,
            captions_are_subtitles=captions_are_subtitles,
            encode_formats=encode_formats,
            detect_cuts=detect_cuts,
            cut_detection_mode=cut_detection_mode,
            cut_framerates=cut_framerates,
            cuts_are_clips=cuts_are_clips,
        )
    elif stage == "subset":
        shard_iterator = OutputSharder(url_list, input_format, done_shards, sampler=sampler)  # type: ignore
        worker = SubsetWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            thread_count=thread_count,
            number_sample_per_shard=number_sample_per_shard,
            oom_shard_count=oom_shard_count,
            encode_formats=encode_formats,
        )
    elif stage == "optical_flow":
        shard_iterator = OutputSharder(url_list, input_format, done_shards, sampler=sampler)  # type: ignore
        if optical_flow_params is None:
            optical_flow_params = {
                "detector": "cv2",
                "detector_args": None,
                "fps": -1,
                "downsample_size": None,
                "dtype": "fp16",
            }
        else:
            optical_flow_dtype = optical_flow_params.get("dtype", None)
            if optical_flow_dtype:
                assert optical_flow_dtype in [
                    "fp16",
                    "fp32",
                ], "please select either fp16 or fp32 for optical flow dtype"
            else:
                optical_flow_params["dtype"] = "fp16"

        is_slurm_task = "GLOBAL_RANK" in os.environ and distributor == "multiprocessing"
        worker = OpticalFlowWorker(  # type: ignore
            sample_writer_class=sample_writer_class,
            output_folder=output_folder,
            thread_count=thread_count,
            number_sample_per_shard=number_sample_per_shard,
            oom_shard_count=oom_shard_count,
            encode_formats=encode_formats,
            optical_flow_params=optical_flow_params,
            is_slurm_task=is_slurm_task,
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")

    print("Starting the downloading of this file")
    called_from_slurm = False
    if distributor == "multiprocessing":
        distributor_fn = multiprocessing_distributor
        called_from_slurm = "GLOBAL_RANK" in os.environ
    elif distributor == "pyspark":
        distributor_fn = pyspark_distributor
    elif distributor == "slurm":
        slurm_args = {"_".join(key.split("_")[1:]): local_args[key] for key in local_args if key.startswith("slurm")}
        worker_args = {key: local_args[key] for key in local_args if not key.startswith("slurm")}
        distributor_fn = SlurmDistributor(worker_args=worker_args, **slurm_args)
    else:
        raise ValueError(f"Distributor {distributor} not supported")

    distributor_fn(
        processes_count,
        worker,
        shard_iterator,
        subjob_size,
        max_shard_retry,
    )
    logger_process.join()
    if not called_from_slurm:
        fs.rm(run_tmp_dir, recursive=True)


def main():
    fire.Fire(video2dataset)


if __name__ == "__main__":
    main()
