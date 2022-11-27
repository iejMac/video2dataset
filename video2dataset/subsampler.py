"""all subsampler for video and audio
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes):
        return video_bytes, None

class ClippingSubsampler:
    def __init__(self):
        pass
    def __call__(self, video_bytes, metadata):
        if "clips" not in metadata:
            return [video_bytes], [metadata], None
        
        # TODO: use meta["key"] to change the key according to clip count
        clip_id = 0

        # TODO: take videos and frament into clips according to metadata["clips"]
        # TODO: change metadata["clips"] to be it's own timeframe
        # TODO: subtitle chopping
        return [video_bytes], [metadata], None
