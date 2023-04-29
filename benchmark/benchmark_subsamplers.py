import os
import itertools
import platform
import psutil
import yaml
from omegaconf import OmegaConf
from webdataset import WebLoader

from video2dataset import subsamplers
from video2dataset.dataloader import get_video_dataset
from subsampler_benchmarks import *

# Add this function to gather system information
def gather_system_info():
    cpu_count = os.cpu_count()
    cpu_info = platform.processor()
    gpu_info = [None]

    # TODO: how do you get GPU info on non linux
    if platform.system() == "Linux":
        try:
            gpu_info = os.popen("nvidia-smi --query-gpu=gpu_name --format=csv,noheader").read().splitlines()
            gpu_count = len(gpu_info)
        except KeyError:
            pass

    return {
        'platform': platform.system(),
        'cpu_count': cpu_count,
        'cpu_info': cpu_info,
        'gpu_info': gpu_info[0],
        'gpu_count': len(gpu_info),
    }


# Add this function to load and parse the config file
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return OmegaConf.create(config)


def create_parameter_grid(params):
    keys, values = zip(*params.items())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]


def main():
    # Gather system information
    system_info = gather_system_info()
    print(f"System Info: {system_info}")

    # Load config
    benchmark_config = load_config('subsamplers_config.yaml')

    # Dataloader
    ds = get_video_dataset(
        urls=benchmark_config.video_set[0].path,
        batch_size=1,
        decoder_kwargs={}, # load bytes
    )

    # Initialize all benchmarkers
    benchmarks = {}
    for subsampler_config in benchmark_config.subsamplers:
        subsampler_name = subsampler_config.name
        params_grid = create_parameter_grid(OmegaConf.to_container(subsampler_config.parameters, resolve=True)[0])
        benchmarks[subsampler_name] = [benchmark_map[subsampler_name](grid) for grid in params_grid]

    for subsampler, benchmarkers in benchmarks.items():
        for sample in ds:
            # TODO: parallelize this in a safe way i.e. each benchmarker gets certain amount of cores (no interference)
            # TODO: report per-core metrics
            [bm(sample) for bm in benchmarkers]

    # TODO: Normalize metrics for core count, sample count, bytes count etc.
    # TODO: visualize in nice way - visual repr but also raw numbers in some json or something

if __name__ == "__main__":
    main()
