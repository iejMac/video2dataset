import pytest
from video2dataset.main import video2dataset


def test_e2e():
    url_list = "tests/test_files/test_webvid.csv"
    video2dataset(
        url_list,
        input_format="csv",
        output_format="webdataset",
        output_folder="tests/dataset",
        url_col="contentUrl",
        caption_col="name",
        save_additional_columns=["videoid","page_idx","page_dir","duration"],
        video_height=360,
        video_width=640,
        number_sample_per_shard=10,
        processes_count=2,
    )
