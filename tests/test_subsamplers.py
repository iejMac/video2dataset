"""test video2dataset subsamplers"""
import os
import pytest
import ffprobe
import ffmpeg
import tempfile


from video2dataset.subsamplers import ClippingSubsampler, get_seconds, ResolutionSubsampler


SINGLE = [[50.0, 60.0]]
MULTI = [
    ["00:00:09.000", "00:00:13.500"],
    ["00:00:13.600", "00:00:24.000"],
    ["00:00:45.000", "00:01:01.230"],
    ["00:01:01.330", "00:01:22.000"],
    ["00:01:30.000", "00:02:00.330"],
]


@pytest.mark.parametrize("clips", [SINGLE, MULTI])
def test_clipping_subsampler(clips):
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")  # video lenght - 2:02
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ClippingSubsampler(3)

    metadata = {
        "key": "000",
        "clips": clips,
    }

    video_fragments, meta_fragments, error_message = subsampler(video_bytes, metadata)
    assert error_message is None
    assert len(video_fragments) == len(meta_fragments) == len(clips)

    for vid_frag, meta_frag in zip(video_fragments, meta_fragments):
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(vid_frag)

            key_ind = int(meta_frag["key"].split("_")[-1])
            s, e = meta_frag["clips"][0]

            assert clips[key_ind] == [s, e]  # correct order

            s_s, e_s = get_seconds(s), get_seconds(e)
            frag_len = get_seconds(ffprobe.FFProbe(tmp.name).metadata["Duration"])

            assert abs(frag_len - (e_s - s_s)) < 5.0  # currently some segments can be pretty innacurate


@pytest.mark.parametrize("dimensions", [(144, 192), (1080, 1920)])
def test_resolution_subsampler(dimensions):
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")  # video lenght - 2:02
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    h, w = dimensions
    subsampler = ResolutionSubsampler(h, w)

    subsampled_videos, error_message = subsampler([video_bytes])
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(subsampled_videos[0])

        probe = ffmpeg.probe(tmp.name)
        video_stream = [stream for stream in probe["streams"] if stream["codec_type"] == "video"][0]
        h_vid, w_vid = video_stream["height"], video_stream["width"]

        assert h == h_vid
        assert w == w_vid
