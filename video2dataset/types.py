"""Type definitions for video2dataset."""
from typing import List, TypedDict


class EncodeFormats(TypedDict, total=False):
    video: str
    audio: str


class Streams(TypedDict, total=False):
    video: List[bytes]
    audio: List[bytes]
