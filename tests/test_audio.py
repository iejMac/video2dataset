import os

import pandas as pd
import pytest
import tempfile
import subprocess
from video2dataset.data_reader import VideoDataReader
import ffmpeg


@pytest.mark.parametrize(
    "input_file, sample_rate",
    [("test_webvid.csv", 16000), ("test_yt.csv", 24000), ("test_yt.csv", 44100), ("test_webvid.csv", 24000)],
)
def test_audio(input_file, sample_rate):
    encode_formats = {"video": "mp4", "audio": "mp3", "sample_rate": sample_rate}
    current_folder = os.path.dirname(__file__)
    url_list = pd.read_csv(os.path.join(current_folder, f"test_files/{input_file}"))["contentUrl"]
    with tempfile.TemporaryDirectory() as tmpdir:
        video_data_reader = VideoDataReader(
            video_size=360, dl_timeout=60, tmp_dir=tmpdir, encode_formats=encode_formats, yt_meta_args=None
        )
        for i, url in enumerate(url_list):
            key, streams, yt_meta_dict, error_message = video_data_reader((i, url))
            aud_bytes = streams.get("audio", None)
            vid_bytes = streams.get("video", None)
            if aud_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
                    f.write(aud_bytes)

                    out = subprocess.check_output(f"file {f.name}".split()).decode("utf-8")
                    assert "Audio file with ID3 version" in out

                    result = ffmpeg.probe(f.name)
                    sr = result["streams"][0]["sample_rate"]

                    assert int(sr) == encode_formats["sample_rate"]

            with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
                f.write(vid_bytes)
                out = subprocess.check_output(f"file {f.name}".split()).decode("utf-8").lower()

                assert "mp4" in out
