import os

import pandas as pd
import pytest
import tempfile
import subprocess
from video2dataset.data_reader import VideoDataReader
import json


@pytest.mark.parametrize("input_file", ["test_webvid.csv", "test_yt.csv"])
def test_audio(input_file):
    encode_formats = {"video": "mp4", "audio": "mp3", "sample_rate": 16000}
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    with tempfile.TemporaryDirectory() as tmpdir:
        video_data_reader = VideoDataReader(
            video_size=360, dl_timeout=60, tmp_dir=tmpdir, encode_formats=encode_formats, yt_meta_args=None
        )
        for i, url in enumerate(url_list):
            key, vid_bytes, aud_bytes, yt_meta_dict, error_message = video_data_reader((i, url))
            if aud_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(aud_bytes)

                    out = subprocess.check_output(f"file {f.name}".split()).decode("utf-8")
                    assert "Audio file with ID3 version" in out

                    result = subprocess.check_output(
                        [
                            "ffprobe",
                            "-hide_banner",
                            "-loglevel",
                            "panic",
                            "-show_format",
                            "-show_streams",
                            "-of",
                            "json",
                            f.name,
                        ]
                    )

                    result = json.loads(result)
                    sr = result["streams"][0]["sample_rate"]

                    assert int(sr) == encode_formats["sample_rate"]

            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(vid_bytes)
                out = subprocess.check_output(f"file {f.name}".split()).decode("utf-8").lower()

                assert "mp4" in out