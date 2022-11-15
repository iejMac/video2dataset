#!/bin/bash

mkdir -p dataset
video2dataset benchmark_vids.parquet --dest="dataset" --output-format="files" --metadata-columns="videoID,title,description,start,end"
