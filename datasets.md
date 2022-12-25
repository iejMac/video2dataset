# Documentation of datasets that have been processed using video2dataset


## [WebVid](https://m-bain.github.io/webvid-dataset/)
Fill this out maybe with data from Robin's run when he does it.


## [ACAV100M](https://acav100m.github.io)
Here's the command used to process the ACAV100M dataset.

```python
video2dataset(
	url_list="/admin/home-iejmac/datasets/acav100m/ACAV100M_clip_unique.parquet",
	output_folder="s3://s-laion/acav100m/mp4_acav100m",
	output_format="webdataset",
	input_format="parquet",
	url_col="videoLoc",
	caption_col="title",
	clip_col="clip",
	save_additional_columns=["description", "videoID", "start", "end"],
	enable_wandb=True,
	video_height=360,
	video_width=640,
	number_sample_per_shard=2000,
	subjob_size=10000,
	processes_count=96,
	thread_count=48,
	distributor="pyspark",
)
```

The job ran on 64 nodes with 48 CPU cores each and processed the dataset in 7 hours. This wasn't the entire ACAV100M dataset but instead a 60M subset of only the unique videos in the dataset. The title and description of the videos were pre-extracted and placed in the parquet file along with the description and also the timespan of the clip was processed into the format that video2dataset expects. [Here's](https://wandb.ai/iejmac/video2dataset/runs/3kon2409?workspace=user-iejmac) a link to the wandb logs from the processing of the dataset. We were able to get 0.8 vid/s/core which comes out to approximately 100MB/s/core.
