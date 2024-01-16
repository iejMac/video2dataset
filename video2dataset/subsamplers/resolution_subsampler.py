"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import ffmpeg
import tempfile
from typing import Literal

from .subsampler import Subsampler


class ResolutionSubsampler(Subsampler):
    """
    Adjusts the resolution of the videos to the specified height and width.

    Args:
        resize_mode (list[str]): List of resize modes to apply. Possible options are:
            scale: scale video keeping aspect ratios (currently always picks video height)
            crop: center crop to video_size x video_size
            pad: center pad to video_size x video_size
        video_size (int): Target resolution of the videos (both height and width).
        height (int): Height of video - video_size will be ignored if this is set.
        width (int): Width of video - video_size will be ignored if this is set.
    """

    def __init__(
        self,
        resize_mode: Literal["scale", "crop", "pad"],
        video_size: int = -1,
        height: int = -1,
        width: int = -1,
    ):
        if height > 0 and width > 0:
            self.height = height
            self.width = width
        elif height > 0 or width > 0:
            return None, None, "Either both height and width must be set or neither - you can also use video_size instead if you want to set both values simultaneously"
        elif video_size > 0:
            self.height = video_size
            self.width = video_size
        else:
            return None, None, "Either video_size or both height and width must be set"
        self.resize_mode = resize_mode

    def __call__(self, streams, metadata=None):
        video_bytes = streams["video"]
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                    if "scale" in self.resize_mode:
                        _ = _.filter("scale", w=self.width, h=self.height)
                    if "crop" in self.resize_mode:
                        _ = _.filter("crop", w=self.width, h=self.height)
                    if "pad" in self.resize_mode:
                        _ = _.filter("pad", w=self.width, h=self.height)
                    _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], None, str(err)

                with open(f"{tmpdir}/output.mp4", "rb") as f:
                    subsampled_bytes.append(f.read())
        streams["video"] = subsampled_bytes
        return streams, None, None
