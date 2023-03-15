"""
Load the data (videos and metadata) into bytes, tensors, strings etc.
"""

from .bytes_dataloader import get_bytes_dataloader
from .train_dataloader import get_video_data  # WIP, currently works for open_clip
from .train_dataloader import get_video_dataset
