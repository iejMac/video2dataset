"""
frame subsampler adjusts the fps of the videos to some constant value
"""
import tempfile
import os
import ffmpeg
from typing import List, Tuple

from video2dataset.types import Metadata, Error, FFmpegStream


class AudioRateSubsampler:
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Args:
        sample_rate (int): Target sample rate of the audio.
        encode_format (str): Format to encode in (i.e. m4a)
    """

    def __init__(self, sample_rate, encode_format, n_audio_channels=None):
        self.sample_rate = sample_rate
        self.encode_format = encode_format
        self.n_audio_channels = n_audio_channels

    def __call__(self, stream: FFmpegStream, tmpdir: str, metadatas: List[Metadata]) -> Tuple[FFmpegStream, List[Metadata], Error]:
        ext = self.encode_format
        ffmpeg_args = {"ar": str(self.sample_rate), "f": ext}
        if self.n_audio_channels is not None:
            ffmpeg_args["ac"] = str(self.n_audio_channels)
        stream = stream.output(f"{tmpdir}/output.{ext}", **ffmpeg_args).run(capture_stdout=True, quiet=True)

        return stream, metadata, None
