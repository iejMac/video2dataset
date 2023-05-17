# [VideoCC](https://github.com/google-research-datasets/videoCC-data)

VideoCC is a video-text dataset with ~10M samples created by starting from an image-text dataset and retrieving videos with frames similar to images in that dataset.

## Download the metadata

Go to [this section](https://github.com/google-research-datasets/videoCC-data#data-format-for-videocc) of their README and download the CSV file. It will need some simple processing which can be done with this code snippet:
```python3
import pandas as pd

df = pd.read_csv("video_cc_public.csv")
df.columns = ["video_url", "start", "end", "caption"]
df["video_url"] = df["video_url"].apply(lambda x: f"https://www.youtube.com/watch?v={x}")
df['start'] = df['start'] / 1_000_000
df['end'] = df['end'] / 1_000_000
df.to_parquet("video_cc.parquet")
```

## Download and process the videos using video2dataset:

The dataset can be downloaded with cut detection and clipping using the following python code:

```python3
from video2dataset import video2dataset

if __name__ == '__main__':

    yt_metadata_args = {
        'writesubtitles': True, # whether to write subtitles to a file
        'subtitleslangs': ['en'], # languages of subtitles (right now support only one language)
        'writeautomaticsub': True, # whether to write automatic subtitles
        'get_info': True # whether to save a video meta data into the output JSON file
    }

    video2dataset(
        url_list='video_cc.parquet',
        input_format='parquet',
        output_format='webdataset',
        output_folder='dataset',
        url_col="video_url",
        number_sample_per_shard=1000,
        processes_count=48,
        thread_count=48,
        yt_metadata_args=yt_metadata_args,
        video_size=360,
        encode_formats={"video": "mp4"},
        stage="download",
        extract_compression_metadata=True,
        clipping_precision="keyframe_adjusted",
        min_clip_length=3,
        max_clip_length=20,
        detect_cuts=True,
        cuts_are_clips=True,
        cut_detector_min_scene_len=15,
        cut_detector_threshold=20,
        cut_detection_mode="all",
        distributor="multiprocessing",
    )
```
