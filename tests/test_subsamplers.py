"""test video2dataset subsamplers"""
import os
import ffprobe
import tempfile


from video2dataset.subsampler import ClippingSubsampler, get_seconds


def test_clipping_subsampler():
    current_folder = os.path.dirname(__file__)
    video = os.path.join(current_folder, "test_files/test_video.mp4")  # video lenght - 2:02
    with open(video, "rb") as vid_f:
        video_bytes = vid_f.read()

    subsampler = ClippingSubsampler(3)
    clips = [
        ["00:00:07.000", "00:00:13.500"],
        ["00:00:13.600", "00:00:40.000"],
        ["00:00:45.000", "00:01:01.230"],
        ["00:01:01.330", "00:01:20.000"],
        ["00:01:40.000", "00:01:50.330"],
    ]
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

            print(frag_len, e_s - s_s)

            assert abs(frag_len - (e_s - s_s)) < 5.0  # currently some segments can be pretty innacurate
