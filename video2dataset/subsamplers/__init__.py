"""
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .clipping_subsampler import ClippingSubsampler, get_seconds
from .noop_subsampler import NoOpSubsampler
