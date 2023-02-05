"""
frame subsampler adjusts the fps of the videos to some constant value
"""


import tempfile
import os
import ffmpeg


class AudioRateSubsampler:
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Args:
        frame_rate (int): Target frame rate of the videos.
    """
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def __call__(self, audio_bytes):
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    # some ffmpeg func
                    pass
                except Exception as err:  # pylint: disable=broad-except
                    return [], str(err)

                # append some stuff to subsampled_bytes
        return subsampled_bytes, None
