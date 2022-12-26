"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import glob
import ffmpeg
import tempfile


class ResolutionSubsampler:
    """
    """

    def __init__(self):
        pass

    def __call__(self, video_bytes, metadata):
        return video_bytes, metadata, None
