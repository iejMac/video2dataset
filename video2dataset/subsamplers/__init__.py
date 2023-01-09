"""
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .clipping_subsampler import ClippingSubsampler, get_seconds
from .frame_subsampler import FrameRateSubsampler
from .noop_subsampler import NoOpSubsampler
from .resolution_subsampler import ResolutionSubsampler
