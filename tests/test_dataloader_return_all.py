"""
Benchmark dataloader speed
"""
import time
import webdataset as wds
# from video2dataset.dataloader import get_bytes_dataloader
from video2dataset.dataloader import get_video_dataset

# Benchmark videos are the WebVid validation split (5000 videos)
SHARDS = "tests/test_files/return_all_test.tar"

def test_return_all_bs1():
    from webdataset import WebLoader

    decoder_kwargs = {"n_frames": 10, "fps": None, "num_threads": 1}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=1,
        decoder_kwargs=decoder_kwargs,
        resize_size=None,
        crop_size=None,
        return_always=True,
        keys_to_remove=['m4a'],
        repeat=1,
        handler = wds.warn_and_continue
    )
    dl = WebLoader(dset, batch_size=None, num_workers=0)

    expected_keys = {'000000014': True,
                     '000000025': True,
                     '000000356': True,
                     '0000008_00001': False,
                     '0000030_00005': False,
                     '0000038_00003': False}
    received_keys = []
    for samp in dl:
        key = samp['__key__'][0]
        received_keys.append(key)
        assert expected_keys[key] == samp['__corrupted__'].item()

    print("Received keys", received_keys)
    print("Expected keys", list(expected_keys))
    assert set(received_keys) == set(expected_keys)

    print('test with bs=1 passed')

def test_return_all_batched():
    from webdataset import WebLoader

    decoder_kwargs = {"n_frames": 10, "fps": None, "num_threads": 1}

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=4,
        decoder_kwargs=decoder_kwargs,
        resize_size=None,
        crop_size=None,
        return_always=True,
        keys_to_remove=['m4a'],
        repeat=1,
        drop_last=False,
        handler = wds.warn_and_continue
    )
    dl = WebLoader(dset, batch_size=None, num_workers=0,drop_last=False)

    expected_keys = {'000000014': True,
                     '000000025': True,
                     '000000356': True,
                     '0000008_00001': False,
                     '0000030_00005': False,
                     '0000038_00003': False}
    received_keys = []
    for samp in dl:
        for i in range(len(samp["__key__"])):
            key = samp['__key__'][i]
            received_keys.append(key)
            assert expected_keys[key] == samp['__corrupted__'][i].item()

    print("Received keys", received_keys)
    print("Expected keys", list(expected_keys))
    assert set(received_keys) == set(expected_keys)

    print('test with bs=4 passed')


if __name__ == "__main__":
    test_return_all_bs1()
    test_return_all_batched()