#!/bin/bash

# video2dataset --url_list="s3://stability-west/acav-w-OF/{000...499}/{000000...000500}.tar" \
# 	--input_format="webdataset" \
# 	--output-format="webdataset" \
# 	--output_folder="s3://stability-west/acav-w-OF-captions/" \
# 	--stage "caption" \
# 	--encode_formats '{"caption": "str"}' \
# 	--config "caption" \

video2dataset --url_list="s3://stability-west/acav/coca_fixed/" \
	--input_format="webdataset" \
	--output-format="webdataset" \
	--output_folder="s3://stability-west/acav/new_captions/" \
	--stage "caption" \
	--encode_formats '{}' \
	--config "/admin/home-iejmac/video2dataset/examples/caption.yaml" \
	--enable_wandb True \
