"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
from typing import Literal
from typing import List, Tuple

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, FFmpegStream


class ResolutionSubsampler(Subsampler):
    """
    Adjusts the resolution of the videos to the specified height and width.
    Please do not set both video_size and height/width. This will result in an error.
    If both height and width are set, scale mode output will have the specified height (ignoring width).

    Args:
        resize_mode (list[str]): List of resize modes to apply. Possible options are:
            scale: scale video keeping aspect ratios
            crop: center crop to video_size x video_size
            pad: center pad to video_size x video_size
        height (int): Height of video.
        width (int): Width of video.
        video_size (int): Both height and width.
        encode_format (str): Format to encode in (i.e. mp4)
    """

    def __init__(
        self,
        resize_mode: Literal["scale", "crop", "pad"],
        height: int = -1,
        width: int = -1,
        video_size: int = -1,
        encode_format: str = "mp4",
    ):
        if video_size > 0 and (height > 0 or width > 0):
            raise ValueError("Either set video_size, or set height and/or width")
        self.resize_mode = resize_mode
        self.height = height if video_size < 0 else video_size
        self.width = width if video_size < 0 else video_size
        self.video_size = video_size
        self.encode_format = encode_format

    def __call__(self, stream: FFmpegStream, tmpdir: str, metadatas: List[Metadata]) -> Tuple[FFmpegStream, List[Metadata], Error]:
        if "scale" in self.resize_mode:
            if self.height > 0:
                stream = stream.filter("scale", -2, self.height)
            else:
                stream = stream.filter("scale", self.width, -2)
        if "crop" in self.resize_mode:
            stream = stream.filter("crop", w=self.width, h=self.height)
        if "pad" in self.resize_mode:
            stream = stream.filter("pad", w=self.width, h=self.height)
        return stream, metadatas, None
