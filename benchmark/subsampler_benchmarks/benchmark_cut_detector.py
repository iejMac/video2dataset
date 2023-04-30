import time

from video2dataset.subsamplers import CutDetectionSubsampler


class BenchmarkResolutionSubsampler(ResolutionSubsampler):
    def __init__(self, subsampler_args):
        super().__init__(**subsampler_args)

        self.subsampler_args = subsampler_args
        self.subsampler_name = "CutDetectionSubsampler"
        # TODO: maybe add "video seconds"
        # TODO: maybe add frames
        self.metrics = {
            "time": 0.0,
            "samples": 0,
            "bytes": 0
        }
    def __call__(self, sample):
        streams = {"video": sample["mp4"]}
        self.metrics["bytes"] += len(streams["video"])

        t0 = time.time()
        out = super().__call__(video_bytes)
        tf = time.time()

        self.metrics["time"] += tf-t0
        self.metrics["samples"] += 1
