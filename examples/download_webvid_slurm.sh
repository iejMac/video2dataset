#!/bin/bash

wget -nc http://www.robots.ox.ac.uk/~maxbain/webvid/results_2M_val.csv

video2dataset --url_list="results_2M_val.csv" \
        --input_format="csv" \
        --processes_count 15 \
        --output-format="webdataset" \
	      --output_folder="dataset" \
        --url_col="contentUrl" \
        --caption_col="name" \
        --save_additional_columns='[videoid,page_idx,page_dir,duration]' \
        --video_size=360 \
        --number_sample_per_shard=100 \
        --slurm_cpus_per_task 15 \
        --slurm_partition "<NODE_NAME>" \
        --slurm_gpus_per_node 0 \
        --number_sample_per_shard 100 \
        --slurm_n_nodes 2 \
        --slurm_account "<YOUR_ACCOUNT>" \
        --slurm_cache_path "<YOUR_HOME>/.slurm_cache" \
        --distributor "slurm"