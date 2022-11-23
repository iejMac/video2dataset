"""all subsampler for video and audio
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes):
        return video_bytes, None
