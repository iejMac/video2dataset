"""
Transformations you might want to apply to data during loading
"""
import math
from abc import abstractmethod
from typing import Optional, Dict, Any, Union

import cv2
import torch
import numpy as np

from .video_decode import PRNGMixin


class VideoResizer(PRNGMixin):
    """Resizes frames to specified height and width"""

    def __init__(
        self,
        size=None,
        crop_size=None,
        random_crop=False,
        key="mp4",
        width_key="original_width",
        height_key="original_height",
    ):
        self.key = key

        self.height_key = height_key
        self.width_key = width_key

        # resize size as [h,w]
        self.resize_size = size
        # crop size as [h,w]
        self.crop_size = crop_size
        if isinstance(self.crop_size, int):
            self.crop_size = [self.crop_size] * 2
        self.random_crop = random_crop and self.crop_size is not None

        if self.crop_size or self.resize_size:
            print(f"{self.__class__.__name__} is resizing video to size {self.resize_size} ...")

            if self.crop_size:
                print(f'... and {"random" if self.random_crop else "center"} cropping to size {self.crop_size}.')
        else:
            print(f"WARNING: {self.__class__.__name__} is not resizing or croppping videos. Is this intended?")

    def _get_rand_reference(self, resize_size, h, w):
        """gets random reference"""
        assert resize_size is None or (
            self.crop_size[0] <= resize_size[0] and self.crop_size[1] <= resize_size[1]
        ), "Resize size must be greater or equal than crop_size"

        # consistent random crop
        min_x = math.ceil(self.crop_size[1] / 2)
        max_x = w - min_x
        if min_x == max_x:
            # catch corner case
            max_x = min(max_x + 1, w)
        min_y = math.ceil(self.crop_size[0] / 2)
        max_y = h - min_y
        if min_y == max_y:
            # catch corner case
            max_y = min(max_y + 1, h)

        try:
            x = self.prng.randint(min_x, max_x, 1).item()
            y = self.prng.randint(min_y, max_y, 1).item()
        except ValueError as e:
            print("Video size not large enough, consider reducing size")
            print(e)
            raise e

        reference = [y, x]
        return reference

    # def _get_resize_size(self, frame, orig_h, orig_w):
    def _get_resize_size(self, frame):
        """gets resize size"""
        orig_h, orig_w = frame.shape[:2]
        if self.resize_size is not None:
            if isinstance(self.resize_size, int):
                f = self.resize_size / min((orig_h, orig_w))
                resize_size = [int(round(orig_h * f)), int(round(orig_w * f))]
            else:
                resize_size = self.resize_size
            h, w = resize_size
        else:
            resize_size = None
            h, w = frame.shape[:2]

        return resize_size, (h, w)

    def _get_reference_frame(self, resize_size, h, w):
        """gets reference frame"""
        if self.random_crop:
            reference = self._get_rand_reference(resize_size, h, w)
        else:
            reference = [s // 2 for s in [h, w]]

        return reference

    def __call__(self, data):
        if self.crop_size is None and self.resize_size is None:
            if isinstance(data[self.key], list):
                # convert to tensor
                data[self.key] = torch.from_numpy(np.stack(data[self.key]))
            return data

        result = []

        if self.key not in data:
            raise KeyError(f"Specified key {self.key} not in data")

        vidkey = self.key
        frames = data[vidkey]

        if isinstance(frames, int):
            raise TypeError(f"Frames is int: {frames}")

        # for videos: take height and width of first frames since the same for all frames anyways,
        # if resize size is integer, then this is used as the new size of the smaller size
        # orig_h = data[self.height_key][0].item()
        # orig_w = data[self.width_key][0].item()

        # resize_size, (h, w) = self._get_resize_size(frames[0], orig_h, orig_w)
        resize_size, (h, w) = self._get_resize_size(frames[0])
        reference = self._get_reference_frame(resize_size, h, w)

        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()

        for frame in frames:

            if resize_size is not None:
                frame = cv2.resize(
                    frame,
                    tuple(reversed(resize_size)),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            if self.crop_size is not None:
                x_ = reference[1] - int(round(self.crop_size[1] / 2))
                y_ = reference[0] - int(round(self.crop_size[0] / 2))

                frame = frame[
                    int(y_) : int(y_) + self.crop_size[0],
                    int(x_) : int(x_) + self.crop_size[1],
                ]

            # TODO: maybe lets add other options for normalization
            # will need for VideoCLIP built on top of CLIP
            frame = frame.astype(float) / 127.5 - 1.0

            frame = torch.from_numpy(frame)
            result.append(frame)

        data[vidkey] = torch.stack(result).to(torch.float16)
        return data


class MetaDataScore:
    """Computes a scalar score based on a given metadata key"""

    def __init__(
        self,
        meta_key: Optional[str] = None,
        encoded_fps: Optional[Union[int, float]] = None,
        video_key: Optional[str] = "mp4",
    ):
        self.meta_key = meta_key
        self.video_key = video_key
        self.encoded_fps = encoded_fps

        if self.meta_key is not None:
            assert self.encoded_fps is not None, "If meta key is not None, encoded_fps also needs to be specified"

    @abstractmethod
    def compute_score(self, metaseq: np.ndarray):
        raise NotImplementedError("Abstract method must not be called")

    @abstractmethod
    def _convert_to_np(self, meta_sequence: Any) -> np.ndarray:
        raise NotImplementedError("Abstract method must not be called")

    def _get_meta_sequence(
        self,
        metadata: np.ndarray,
        start_frame: int,
        native_fps: float,
        video_fps: Union[float, int],
        n_video_frames: int,
        original_n_frames: int,
    ) -> np.ndarray:
        """
        Extracts the sub sequence from the metadata sequence which has the same timespan
        than the loaded video clip
        """
        assert (
            self.encoded_fps <= native_fps
        ), "Encoded fps of metadata must not be larger than native fps, this can only be an error"
        # get strides for meta data and video frames
        meta_stride = int(np.round(native_fps / self.encoded_fps))
        video_stride = int(np.round(native_fps / video_fps))

        # extract meta frames
        available_meta_frames = np.arange(start=0, stop=original_n_frames, step=meta_stride)
        # start and end of video
        end_frame = start_frame + video_stride * n_video_frames

        used_meta_frames = np.logical_and(available_meta_frames >= start_frame, available_meta_frames <= end_frame)

        if not np.any(used_meta_frames):
            used_idx = (np.abs(available_meta_frames - (start_frame + end_frame) / 2)).argmin()
            used_meta_frames = np.asarray(
                [
                    used_idx,
                ]
            )

        return metadata[used_meta_frames]

    def __call__(self, sample: Dict):
        if self.meta_key is None:
            return sample

        start_frame = sample["start_frame"]
        native_fps = sample["native_fps"]
        fps = sample["fps"]
        original_n_frames = sample["original_n_frames"]
        n_frames = len(sample[self.video_key])
        meta_sequence = self._convert_to_np(sample[self.meta_key])

        meta_sequence = self._get_meta_sequence(
            meta_sequence, start_frame, native_fps, fps, n_frames, original_n_frames
        )
        # overwrite metadata sequence with score (since sequence might
        # cause trouble during batching due to varying lengths
        sample[self.meta_key] = self.compute_score(meta_sequence)
        return sample


class MotionScore(MetaDataScore):
    def _convert_to_np(self, meta_sequence: Any) -> np.ndarray:
        return meta_sequence

    def compute_score(self, metaseq: np.ndarray):
        return torch.Tensor([np.mean(metaseq)])


class CutsAdder:
    """Adds cuts to video sample"""

    def __init__(self, cuts_key, video_key="mp4"):
        self.cuts_key = cuts_key
        self.video_key = video_key

    def __call__(self, sample):
        assert self.cuts_key in sample, f'no field with key "{self.cuts_key}" in sample, but this is required.'
        assert self.video_key in sample, f'no field with key "{self.video_key}" in sample, but this is required.'
        sample[self.video_key] = {
            self.video_key: sample[self.video_key],
            self.cuts_key: sample[self.cuts_key],
        }
        del sample[self.cuts_key]
        return sample


class CustomTransforms:
    def __init__(self, key_transform_dict):
        self.key_transform_dict = key_transform_dict

    def __call__(self, sample):
        for key, transform in self.key_transform_dict.items():
            sample[key] = transform(sample[key])
        return sample
