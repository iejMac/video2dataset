import time

from video2dataset.subsamplers import ResolutionSubsampler


class BenchmarkResolutionSubsampler(ResolutionSubsampler):
    def __init__(self, subsampler_args):
        super().__init__(**subsampler_args)

        self.subsampler_args = subsampler_args
        # TODO: maybe add "video seconds"
        # TODO: maybe add frames
        self.metrics = {
            "time": 0.0,
            "samples": 0,
            "bytes": 0
        }
    def __call__(self, sample):
        video_bytes = [sample["mp4"]]
        self.metrics["bytes"] += len(video_bytes[0])

        t0 = time.time()
        out = super().__call__(video_bytes)
        tf = time.time()

        self.metrics["time"] += tf-t0
        self.metrics["samples"] += 1
        print(self.metrics)
