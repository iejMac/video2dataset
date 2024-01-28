"""No operation subsampler"""
from typing import List, Tuple

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, TempFilepaths


class NoOpSubsampler(Subsampler):
    def __init__(self):
        pass

    def __call__(self, filepaths: TempFilepaths, metadata: Metadata) -> Tuple[TempFilepaths, List[Metadata], Error]:
        return filepaths, [metadata], None
