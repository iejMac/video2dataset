#!/bin/bash

video2dataset --url_list="s3://stability-west/acav-w-OF/" \
        --input_format="webdataset" \
        --output-format="webdataset" \
	--output_folder="acav-w-OF-captions" \
	--stage "caption" \
	--encode_formats '{"caption": "str"}' \
	--config "caption" \
