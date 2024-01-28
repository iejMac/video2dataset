"""No operation subsampler"""
from typing import List, Tuple

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, FFmpegStream


class NoOpSubsampler(Subsampler):
    def __init__(self):
        pass

    def __call__(self, stream: FFmpegStream, tmpdir: str, metadatas: List[Metadata]) -> Tuple[FFmpegStream, List[Metadata], Error]:
        return stream, metadatas, None
