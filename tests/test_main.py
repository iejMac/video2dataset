"""end2end test"""
import os

import pandas as pd
import pytest
import tarfile
import tempfile

from video2dataset.main import video2dataset


@pytest.mark.parametrize("input_file", ["test_webvid.csv", "test_yt.csv"])
def test_e2e(input_file):
    current_folder = os.path.dirname(__file__)
    url_list = os.path.join(current_folder, f"test_files/{input_file}")

    sample_count = len(pd.read_csv(url_list))

    with tempfile.TemporaryDirectory() as tmpdir:
        samples_per_shard = 10 if "webvid" in input_file else 3

        video2dataset(
            url_list,
            output_folder=tmpdir,
            input_format="csv",
            output_format="webdataset",
            url_col="contentUrl",
            caption_col="name",
            save_additional_columns=["videoid"],
            video_size=360,
            number_sample_per_shard=samples_per_shard,
            processes_count=1,
            encode_formats={"video": "mp4"},
        )

        for shard in ["00000", "00001"] if sample_count / samples_per_shard > 1.0 else ["00000"]:
            for ext in ["mp4", "json", "txt"]:
                assert (
                    len([x for x in tarfile.open(tmpdir + f"/{shard}.tar").getnames() if x.endswith(f".{ext}")])
                    == samples_per_shard
                )


@pytest.mark.parametrize("input_file", ["test_yt.csv"])
def test_only_subs(input_file):
    current_folder = os.path.dirname(__file__)
    url_list = os.path.join(current_folder, f"test_files/{input_file}")

    sample_count = len(pd.read_csv(url_list))

    with tempfile.TemporaryDirectory() as tmpdir:
        samples_per_shard = 3

        yt_metadata_args = {
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "writeautomaticsub": True,
            "get_info": False,
            "subtitles_clipping": False,
        }

        video2dataset(
            url_list,
            output_folder=tmpdir,
            input_format="csv",
            output_format="webdataset",
            url_col="contentUrl",
            caption_col="name",
            number_sample_per_shard=samples_per_shard,
            processes_count=1,
            yt_metadata_args=yt_metadata_args,
        )

        for shard in ["00000", "00001"] if sample_count / samples_per_shard > 1.0 else ["00000"]:
            for ext in ["json", "txt"]:
                assert (
                    len([x for x in tarfile.open(tmpdir + f"/{shard}.tar").getnames() if x.endswith(f".{ext}")])
                    == samples_per_shard
                )
