"""test video2dataset downloaders"""
import os
import pytest
import ffmpeg


from video2dataset.data_reader import YtDlpDownloader, Mp4Downloader


YT_URL = "https://www.youtube.com/watch?v=jLX0D8qQUBM"
MP4_URL = "https://ak.picdn.net/shutterstock/videos/1053841541/preview/stock-footage-travel-blogger-shoot-a-story-on-top-of-mountains-young-man-holds-camera-in-forest.mp4"


@pytest.mark.parametrize("video_size", [361, 1080])
def test_yt_downloader(video_size):
    ytdlp_downloader = YtDlpDownloader(
        tmp_dir="/tmp", metadata_args=None, video_size=video_size, encode_formats={"video": "mp4"}
    )

    path, aud_path, yt_meta_dict = ytdlp_downloader(YT_URL)

    probe = ffmpeg.probe(path)
    video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
    height = int(video_stream["height"])

    assert height == 480
    os.remove(path)


def test_mp4_downloader():
    mp4_downloader = Mp4Downloader(timeout=10, tmp_dir="/tmp", encode_formats={"video": "mp4"})

    path, aud_path = mp4_downloader(MP4_URL)

    with open(path, "rb") as f:
        assert len(f.read()) > 0
    os.remove(path)
