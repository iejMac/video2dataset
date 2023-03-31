# [HDVILA-100M](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)
HDVILA 100M is a dataset of 100M high-resolution videos from YouTube.

## Download the metdata
First, run `wget -O hdvila100m.zip https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip?sp=r&st=2022-06-28T03:33:11Z&se=2026-01-01T11:33:11Z&spr=https&sv=2021-06-08&sr=b&sig=VaqQkLFDqKinfkaPNs1jJ1EQIYCB%2FUPYiqFqmjWye6Y%3D` to download the HD VILA 100M metadata. Next, just run `unzip hdvilla100m.zip` in order to unzip the metadata. You should now have an `hdvila100m/` directory.

Next, we need to do some preprocessing to get this metadata formatted into a nice parquet. The following script will take the downloaded metadata `.jsonl` files and create a parquet with all the relevant information.

```python
import pandas as pd
import glob
import json
import os
import time
from datetime import datetime

def time_string_to_seconds(timestamp):
    hh,mm,s = timestamp.split(':')
    ss,ms = s.split('.')
    time = 3600*int(hh) +  60*int(mm) + int(ss) + int(ms)/1000
    return time

def convert_clip_list(clip_list):
    return [[time_string_to_seconds(x) for x in clip] for clip in clip_list]

parquet_dir = "/path/to/my/metadata/dir/"

data = []
for jsonl in sorted(glob.glob(f"{parquet_dir}*.jsonl")):
    path = os.path.join(parquet_dir, jsonl)
    with open(path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            clips = [
                json_obj['clip'][i]['span']
                for i in range(len(json_obj['clip']))
            ]

            out = {
                'video_id': json_obj['video_id'],
                'url': json_obj['url'],
                'clips': clips
            }
            data.append(out)

df = pd.DataFrame(data)
df['clips'] = df['clips'].map(lambda x: convert_clip_list(x))
df.to_parquet("hd_vila.parquet")
```

Once you run this, you should have a file `hd_vila.parquet` with all the relevant metadata.

## Download the videos with video2dataset

This command runs video2dataset on the input parquet and saves the video clips along with the metadata in the webdataset format.

```python

# args specify we want to get all metadata from the video and save to the json component of the sample
# this includes the title and description of the video
yt_metadata_args = {
    "get_info": True,  # whether to save a video meta data into the output JSON file
}

video2dataset(
    url_list="/path/to/my/hd_vila.parquet",
    output_folder="/path/to/my/output",
    output_format="webdataset",
    input_format="parquet",
    url_col="url",
    clip_col="clips",
    save_additional_columns=None,
    enable_wandb=True,
    video_size=360,
    number_sample_per_shard=2000,
    subjob_size=10000,
    processes_count=96,
    thread_count=48,
    distributor="multiprocessing",
    yt_metadata_args=yt_metadata_args
)
```
