"""all subsampler for video and audio
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes):
        return video_bytes, None

class ClippingSubsampler:
    def __init__(self, oom_clip_count):
        self.oom_clip_count = oom_clip_count
    def __call__(self, video_bytes, metadata):
        if "clips" not in metadata:
            return [video_bytes], [metadata], None

        clips = metadata.pop("clips")
        video_clips, metadata_clips = [], []
        for clip_id, clip in enumerate(clips):
            video_fragment = video_bytes # TODO take videos and fragment into clips using ffmpeg
            video_clips.append(video_fragment)

            clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                clip_id=clip_id, oom_clip_count=self.oom_clip_count
            ) 

            meta_clip = metadata.copy()
            meta_clip["clips"] = [clip] # set the timeframe of this clip
            meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"
            metadata_clips.append(meta_clip)
        
        # TODO: subtitle chopping
        return video_clips, metadata_clips, None
