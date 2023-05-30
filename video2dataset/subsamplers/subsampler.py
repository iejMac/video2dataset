"""Base subsampler class"""


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    def __init__(self):
        raise NotImplementedError

    def __call__(self, streams, metadata):
        raise NotImplementedError
