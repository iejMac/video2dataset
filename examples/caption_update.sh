#!/bin/bash

video2dataset --url_list="s3://stability-west/video-cc-split-full/"  \
	--input_format="webdataset" \
	--output-format="webdataset" \
	--output_folder="s3://stability-west/video-cc-split-full-captions-globfix-v2/" \
	--stage "caption" \
	--encode_formats '{}' \
	--config "caption" \
	--enable_wandb True \
