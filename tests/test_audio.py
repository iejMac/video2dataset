import os

import pandas as pd
import pytest
import tempfile
import subprocess
from video2dataset.data_reader import VideoDataReader
import ffmpeg


@pytest.mark.parametrize(
    "input_file",
    ["test_webvid.csv", "test_yt.csv"],
)
def test_audio(input_file):
    encode_formats = {"video": "mp4", "audio": "mp3"}
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    with tempfile.TemporaryDirectory() as tmpdir:
        video_data_reader = VideoDataReader(
            video_size=360, dl_timeout=60, tmp_dir=tmpdir, encode_formats=encode_formats, yt_meta_args=None
        )
        for i, url in enumerate(url_list):
            _, streams, _, _ = video_data_reader((i, url))
            aud_bytes = streams.get("audio", None)
            vid_bytes = streams.get("video", None)
            if aud_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(aud_bytes)

                    out = subprocess.check_output(f"file {f.name}".split()).decode("utf-8")
                    assert "Audio file with ID3 version" in out
