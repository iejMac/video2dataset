# [WebVid](https://m-bain.github.io/webvid-dataset/)

The entirety of WebVid can be downloaded very easily using scripts/configs provided in video2dataset/examples. You don't even need to use complex distribution strategies, multiprocessing is fine to download all 10M samples in a timely manner on a single machine.

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
    processes_count: 16
    thread_count: 16
    subjob_size: 1000
    distributor: "multiprocessing"
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

## Performance



