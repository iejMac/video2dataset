# [WebVid](https://m-bain.github.io/webvid-dataset/)

The entirety of WebVid can be downloaded very easily using scripts/configs provided in video2dataset/examples.

## Create the config:

```yaml
subsampling: {}

reading:
    yt_args:
        download_size: 360
        download_audio_rate: 44100
        yt_metadata_args: null
    timeout: 60
    sampler: null

storage:
    number_sample_per_shard: 1000
    oom_shard_count: 5
    captions_are_subtitles: False

distribution:
    processes_count: 32
    thread_count: 32
    subjob_size: 1000
    distributor: "slurm"
    distributor_args:
        cpus_per_task: 64
        partition: "cpu64"
        n_nodes: 50
        account: "laion"
        cache_path: "/fsx/home-iejmac/.slurm_cache"
```

## Download WebVid:

```bash
#!/bin/bash

wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_10M_train.csv

video2dataset --url_list="results_10M_train.csv" \
        --input_format="csv" \
        --output-format="webdataset" \
	--output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --enable_wandb=True \
	--config="path/to/config.yaml" \
```
