"""Base subsampler class"""
from abc import abstractmethod
from typing import List, Tuple

from video2dataset.types import Metadata, Error, FFmpegStream


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, stream: FFmpegStream, tmpdir: str, metadatas: List[Metadata]) -> Tuple[FFmpegStream, List[Metadata], Error]:
        raise NotImplementedError("Subsampler should not be called")
