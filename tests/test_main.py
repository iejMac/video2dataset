"""end2end test"""
import os

import pandas as pd
import pytest
import tarfile
import tempfile

from video2dataset.main import video2dataset


def test_e2e():
    current_folder = os.path.dirname(__file__)
    url_list = os.path.join(current_folder, "test_files/test_webvid.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        samples_per_shard = 10

        video2dataset(
            url_list,
            output_folder=tmpdir,
            input_format="csv",
            output_format="webdataset",
            url_col="contentUrl",
            caption_col="name",
            save_additional_columns=["videoid", "page_idx", "page_dir", "duration"],
            video_height=360,
            video_width=640,
            number_sample_per_shard=samples_per_shard,
            processes_count=1,
        )

        for shard in ["00000", "00001"]:
            for ext in ["mp4", "json", "txt"]:
                assert (
                    len([x for x in tarfile.open(tmpdir + f"/{shard}.tar").getnames() if x.endswith(f".{ext}")])
                    == samples_per_shard
                )
