"""No operation subsampler"""
from ffmpeg.nodes import Node
from typing import List, Tuple

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error


class NoOpSubsampler(Subsampler):
    def __init__(self):
        pass

    def __call__(self, ffmpeg_node: Node, tmpdir: str, metadatas: List[Metadata]) -> Tuple[Node, List[Metadata], Error]:
        return filepaths, metadatas, None
