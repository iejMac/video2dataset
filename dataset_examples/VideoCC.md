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
```

## Download and process the videos using video2dataset:





