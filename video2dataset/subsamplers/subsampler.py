"""Base subsampler class"""
from abc import abstractmethod
from ffmpeg.nodes import Node
from typing import List, Tuple

from video2dataset.types import Metadata, Error


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, ffmpeg_node: Node, tmpdir: str, metadatas: List[Metadata]) -> Tuple[Node, List[Metadata], Error]:
        raise NotImplementedError("Subsampler should not be called")
