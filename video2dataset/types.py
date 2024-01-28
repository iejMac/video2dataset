"""Type definitions for video2dataset."""
from typing import List, TypedDict, Optional


class EncodeFormats(TypedDict, total=False):
    video: str
    audio: str


class Streams(TypedDict, total=False):
    video: List[bytes]
    audio: List[bytes]


# TODO: make more structured
Metadata = dict


Error = Optional[str]


# TODO: remove after refactoring is complete
class TempFilepaths(TypedDict, total=False):
    video: List[str]
    audio: List[str]
