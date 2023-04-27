import time
from video2dataset.subsamplers import OpticalFlowSubsampler
from video2dataset.dataloader import get_video_dataset
from webdataset import WebLoader

shards = "s3://stability-west/hd-vila/high_res_cut_detection_us_west_cpu128/00903.tar"
import numpy as np

detector = "cv2"
fps = 2
downsample_size = 16
dtype = "fp16"
detector_args = None

optical_flow_subsampler = OpticalFlowSubsampler(
    detector=detector,
    args=detector_args,
    dtype=dtype,
    is_slurm_task=False,
)

decoder_kwargs = {
    "n_frames": None,
    "fps": fps,
    "num_threads": 8,
    "return_bytes": True,
}

dset = get_video_dataset(
    urls=shards,
    batch_size=1,
    decoder_kwargs=decoder_kwargs,
    resize_size=downsample_size,
    crop_size=None,
    enforce_additional_keys=[],
)

count = 0
t_optical_flow = 0
t_decode = 0
t_total = 0

t0_decode = time.time()
t0_total = time.time()
for sample in dset:
    t1_decode = time.time()

    count += 1
    frames = np.array(sample.get("mp4")[0]).astype(np.float32)
    rescale_factor = sample.get("rescale_factor")[0]

    t0_cv2 = time.time()
    optical_flow, metrics, error_message = optical_flow_subsampler(frames, rescale_factor)
    t1_cv2 = time.time()
    t1_total = time.time()

    t_optical_flow += t1_cv2 - t0_cv2
    t_decode += t1_decode - t0_decode
    t0_decode = t1_decode
    t_total += t1_total - t0_total
    t0_total = t1_total
    if count % 10 == 1:
        print("=" * 100)
        print(f"Decoder throughput", count, "samples in", t_decode, "seconds, or", count / t_decode, "samples/s")
        print(
            f"Optical flow throughput:",
            count,
            "samples in",
            t_optical_flow,
            "seconds, or",
            count / t_optical_flow,
            "samples/s",
        )
        print(f"Total throughput:", count, "samples in", t_total, "seconds, or", count / t_total, "samples/s")
        print("=" * 100)

print(count / t_optical_flow)
