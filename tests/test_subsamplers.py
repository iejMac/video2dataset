"""test video2dataset subsamplers"""
import os

from video2dataset.subsampler import ClippingSubsampler, get_seconds

def test_clipping_subsampler():
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4") # video lenght - 2:02
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ClippingSubsampler(3)
    clips = [
        ["00:00:03.120", "00:00:13.500"],
        ["00:00:13.600", "00:00:40.000"],
        ["00:00:45.000", "00:01:01.230"],
        ["00:01:01.330", "00:01:20.000"],
        ["00:01:40.000", "00:01:50.330"],
    ]
    metadata = {
        "key": "000",
        "clips": clips,
    }

    vid_fragments, meta_fragments, error_message = subsampler(video_bytes, metadata)

    assert len(vid_fragments) == len(meta_fragments) == len(clips)
