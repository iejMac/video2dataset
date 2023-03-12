"""
Transformations you might want to apply to data during loading
"""
from typing import Optional, Tuple

try:
    import torch

    from torchvision.transforms import (
        Normalize,
        Compose,
        RandomResizedCrop,
        InterpolationMode,
        ToTensor,
        Resize,
        CenterCrop,
        ToPILImage,
    )

    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
except ModuleNotFoundError as e:
    OPENAI_DATASET_MEAN, OPENAI_DATASET_STD = None, None


def video_transform(
    frame_size: int,
    n_frames: int,
    take_every_nth: int,
    is_train: bool,
    frame_mean: Optional[Tuple[float, ...]] = None,
    frame_std: Optional[Tuple[float, ...]] = None,
):
    """simple frame transformations"""

    frame_mean = frame_mean or OPENAI_DATASET_MEAN
    if not isinstance(frame_mean, (list, tuple)):
        frame_mean = (frame_mean,) * 3

    frame_std = frame_std or OPENAI_DATASET_STD
    if not isinstance(frame_std, (list, tuple)):
        frame_std = (frame_std,) * 3

    normalize = Normalize(mean=frame_mean, std=frame_std)

    if is_train:
        transforms = [
            ToPILImage(),
            RandomResizedCrop(
                frame_size,
                scale=(0.9, 0.1),
                interpolation=InterpolationMode.BICUBIC,
            ),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ]
    else:
        transforms = [
            ToPILImage(),
            Resize(frame_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(frame_size),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ]

    frame_transform = Compose(transforms)

    def apply_frame_transform(sample):
        video, _, _ = sample
        video = video.permute(0, 3, 1, 2)

        video = video[::take_every_nth]
        video = video[:n_frames]  # TODO: maybe make this middle n frames

        # TODO: maybe padding isn't the way to go
        # TODO: also F.pad is acting up for some reason
        # isn't letting me input a len 8 tuple for 4d tnesor???
        # video = F.pad(video, tuple([0, 0]*len(video.shape[-3:]) + [0, n_frames - video.shape[0]]))
        if video.shape[0] < n_frames:
            padded_video = torch.zeros(n_frames, *video.shape[1:])
            padded_video[: video.shape[0]] = video
            video = padded_video

        # TODO: this .float() is weird, look how this is done in other places
        return torch.cat([frame_transform(frame.float())[None, ...] for frame in video])

    return apply_frame_transform
