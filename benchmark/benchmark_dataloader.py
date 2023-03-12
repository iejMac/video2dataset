"""
Benchmark dataloader speed
"""
import time
from video2dataset.dataloader import get_bytes_dataloader

# Benchmark videos are the WebVid validation split (5000 videos)
SHARDS = "examples/dataset/{00000..00004}.tar"


def benchmark_bytes_dl():
    # 326.06278216697706
    dl = get_bytes_dataloader(SHARDS, 12)

    count = 0
    t0 = time.time()
    for samp in dl:
        key, vb, cap, meta = samp
        count += 1
    tf = time.time()
    return count / (tf - t0)


if __name__ == "__main__":
    print("Benchmarking bytes dataloader...")
    print(benchmark_bytes_dl())
