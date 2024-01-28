"""Type definitions for video2dataset."""
from ffmpeg.nodes import FilterableStream
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


# this is here because ffmpeg objects aren't type annotated correctly
class FFmpegStream(FilterableStream):
    def filter(self, *args, **kwargs) -> FFmpegStream:
        ...
