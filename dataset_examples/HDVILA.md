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

## Stage 1 (Downloading + Cut Detection)

This command runs video2dataset on the input parquet and saves the videos along with the metadata in the webdataset format.

```python
from video2dataset import video2dataset

if __name__ == '__main__':

    yt_metadata_args = {
        'writesubtitles': True, # whether to write subtitles to a file
        'subtitleslangs': ['en'], # languages of subtitles (right now support only one language)
        'writeautomaticsub': True, # whether to write automatic subtitles
        'get_info': True # whether to save a video meta data into the output JSON file
    }

    video2dataset(
        url_list='/fsx/proj-stablediffusion/datasets/hd-vila/hd_vila.parquet',
        input_format='parquet',
        output_format='webdataset',
        output_folder='s3://stability-west/hd-vila/base_dataset',
        url_col="url",
        enable_wandb=True,
        number_sample_per_shard=100,
        subjob_size=10000,
        processes_count=32,
        thread_count=32,
        detect_cuts=True,
        cut_detection_mode="all",
        yt_metadata_args=yt_metadata_args,
        encode_formats={"video": "mp4", "audio": "m4a"},
        slurm_partition="cpu64",
        slurm_gpus_per_node=0,
        slurm_n_nodes=16,
        slurm_account="laion",
        slurm_cache_path="/fsx/home-iejmac/.slurm_cache",
        distributor="slurm",
    )
```

## Stage 1 Performance

Over the course of downloading we ran into some bugs (which are now resolved) which required restarts, here's the WandB report for [one of the restarts](https://api.wandb.ai/links/iejmac/nn9hcaol). The downloading was performed on a cluster of 16 c6i cpu64 aws nodes (64 cpu cores) and the total downloading time can be deduced from the videos/s in the report (~28 hours). We were not just downlading the videos but also the audio and the youtube metadata and performing cut detection on the videos which is placed in the metadata of the samples.
