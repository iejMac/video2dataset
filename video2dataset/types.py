from typing import List, TypedDict


class EncodeFormats(TypedDict):
    video: str
    audio: str


class Streams(TypedDict):
    video: List[bytes]
    audio: List[bytes]
