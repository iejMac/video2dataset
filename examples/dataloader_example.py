from webdataset import WebLoader

from video2dataset.dataloader import get_video_dataset

# WebVid validation split
SHARDS = "dataset/{00000..00004}.tar"

if __name__ == "__main__":

    decoder_kwargs = {
        "n_frames": 8, # get 8 frames from each video
        "fps": 10, # downsample to 10 FPS
        "num_threads": 12 # use 12 threads to decode the video
    }
    resize_size = crop_size = 256
    batch_size = 32

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
    )

    num_workers = 6 # 6 dataloader workers

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)

    for sample in dl:
        print(sample.keys())







