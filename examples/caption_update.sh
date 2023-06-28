#!/bin/bash

video2dataset --url_list="s3://stability-west/stable-video-dataset-merged/"  \
	--input_format="webdataset" \
	--output-format="webdataset" \
	--output_folder="s3://stability-west/stable-video-dataset-merged-captions-v3/" \
	--stage "caption" \
	--encode_formats '{}' \
	--config "caption" \
	--enable_wandb True \


