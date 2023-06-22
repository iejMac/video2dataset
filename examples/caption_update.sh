#!/bin/bash

# video2dataset --url_list="s3://stability-west/acav-w-OF/{000...499}/{000000...000500}.tar" \
# 	--input_format="webdataset" \
# 	--output-format="webdataset" \
# 	--output_folder="s3://stability-west/acav-w-OF-captions/" \
# 	--stage "caption" \
# 	--encode_formats '{"caption": "str"}' \
# 	--config "caption" \


video2dataset --url_list="/fsx/Andreas/stable-datasets/test_dataset/000/{000000..000009}.tar" \
	--input_format="webdataset" \
	--output-format="webdataset" \
	--output_folder="test-captions/" \
	--stage "caption" \
	--encode_formats '{"caption": "str"}' \
	--config "caption" \
