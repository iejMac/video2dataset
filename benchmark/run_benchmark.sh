#!/bin/bash

rm -rf dataset

video2dataset --url_list="benchmark_vids.parquet" \
--input_format="parquet" \
--output_folder="dataset" \
--output-format="files" \
--url_col="videoLoc" \
--caption_col="title" \
--save_additional_columns='[videoID,description,start,end]' \
--enable_wandb=True
