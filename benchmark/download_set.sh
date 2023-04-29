#!/bin/bash

rm -rf dataset

video2dataset --url_list="video_sets/youtube.parquet" \
        --input_format="parquet" \
        --output_folder="dataset" \
        --output-format="webdataset" \
        --url_col="videoLoc" \
        --caption_col="title" \
        --enable_wandb=True \
        --video_size=360 \
        --number_sample_per_shard=10 \
        --processes_count 10  \
	--enable_wandb False \
