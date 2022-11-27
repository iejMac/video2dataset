"""all subsampler for video and audio
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""
import time
import ffmpeg
import tempfile


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes, metadata):
        return [video_bytes], [metadata], None

class ClippingSubsampler:
    def __init__(self, oom_clip_count):
        self.oom_clip_count = oom_clip_count
    def __call__(self, video_bytes, metadata):
        clips = metadata.pop("clips")
        video_clips, metadata_clips = [], []

        t0 = time.time()
        # TODO: we need to put the extension into the metadata
        # TODO: This can be done better using pipes I just don't feel like sinking too much time into this rn
        video_bytes_file = tempfile.NamedTemporaryFile(suffix=".mp4")
        video_bytes_file.write(video_bytes)
        try:
            for clip_id, clip in enumerate(clips):
                s, e = clip
                # NOTES:
                # - can't do format="mp4" because mp4 expects read_only medium
                # - flv works and you can write as .mp4 but cv2 frame count showed one less frame than if you directly write to mp4...

                # TODO: make much faster, right now it takes up to a few seconds per video since you're reading and outputting every time you want to make a clip

                video_fragment, _ = (
                    ffmpeg.input(video_bytes_file.name)
                    .trim(start=s, end=e)
                    .output("pipe:", format="flv", pix_fmt="rgb24", loglevel="error") # TODO: investigate this flv format more
                    .run(capture_stdout=True)
                )

                video_clips.append(video_fragment)

                clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                    clip_id=clip_id, oom_clip_count=self.oom_clip_count
                ) 

                meta_clip = metadata.copy()
                meta_clip["clips"] = [clip] # set the timeframe of this clip
                meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"
                metadata_clips.append(meta_clip)

            video_bytes_file.close()
            tf = time.time() 
            print(f"Trimming {len(clips)} clips took {tf-t0}")

            # TODO: subtitle chopping
            return video_clips, metadata_clips, None
        except Exception as err:  # pylint: disable=broad-except
            return [], [], str(err)
