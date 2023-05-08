"""No operation subsampler"""


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, streams, metadata):
        return streams, [metadata], None
