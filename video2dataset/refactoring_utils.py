"""Temporary functions used during refactoring"""
import os
import tempfile
from typing import List
import uuid

from video2dataset.types import Streams


def stream_to_temp_filepaths(streams: Streams) -> dict[str, List[str]]:
    """
    This is a temporary workaround for now, while refactoring some of the subsamplers.
    Each subsampler currently works by taking in streams, writing to temp input files, and running ffmpeg to produce temp output files.
    It would be faster to build one combined ffmpeg pipe for all subsamplers and run it only once.
    That way we can simply pass in input filenames, and write final output files directly, without IO on intermediate temp files.
    It's difficult to change everything at once, so we're going to refactor one subsampler at a time.
    This function allows us to save temp files, to give us filenames for passing into refactored subsamplers.
    It would be much better to just start with filepaths to begin with, but this requires big changes to how input streams are processed.
    """

    stream_temp_filepaths = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for modality in streams:
            stream_temp_filepaths[modality] = []
            for stream in streams[modality]:
                stream_uuid = str(uuid.uuid4())
                stream_temp_filepath = os.path.join(tmpdir, stream_uuid)
                with open(stream_temp_filepath, "wb") as f:
                    f.write(stream)
                stream_temp_filepaths[modality].append(stream_temp_filepath)
    return stream_temp_filepaths


def temp_filepaths_to_streams(stream_temp_filepaths: dict[str, List[str]]) -> Streams:
    """Going the other way. Once again, a temporary workaround during refactoring."""

    streams: Streams = {}
    for modality in stream_temp_filepaths:
        streams[modality] = []
        for stream_temp_filepath in stream_temp_filepaths[modality]:
            with open(stream_temp_filepath, "rb") as f:
                streams[modality].append(f.read())
            os.remove(stream_temp_filepath)
    return streams
