"""No operation subsampler"""


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes, metadata):
        return [video_bytes], [metadata], None
