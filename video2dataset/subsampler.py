"""all subsampler for video and audio
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""
import os
import glob
import ffmpeg
import tempfile

from datetime import datetime


class NoOpSubsampler:
    def __init__(self):
        pass

    def __call__(self, video_bytes, metadata):
        return [video_bytes], [metadata], None


def get_seconds(t):
    time_format = "%H:%M:%S.%f"  # TODO: maybe paramaterize this?
    t_obj = datetime.strptime(t, time_format).time()
    return t_obj.second + t_obj.microsecond / 1e6 + t_obj.minute * 60 + t_obj.hour * 3600


class ClippingSubsampler:
    """
    Cuts videos up into segments according to the 'clips' metadata

    expects:
    - clips to be sorted in increasing order and non-overlapping
    - time to be in the format "%H:%M:%S.%f"
    """

    def __init__(self, oom_clip_count):
        self.oom_clip_count = oom_clip_count

    def __call__(self, video_bytes, metadata):
        clips = metadata.pop("clips")

        s_0, e_f = get_seconds(clips[0][0]), get_seconds(clips[-1][1])

        ind = 1
        # we assume there's always one clip which we want to take
        _, e_p = clips[0]
        e_p = get_seconds(e_p)
        splits = [e_p]
        # list of indicies of clips to take, used to discard non-contiguous sections
        take_inds = [0]

        # TODO: make nicer
        for s, e in clips[1:]:
            s, e = get_seconds(s), get_seconds(e)

            if s - e_p <= 1.0:  # no one needs 1.0 second clips + creates less files
                splits += [e] if e != e_f else []
                take_inds.append(ind)
            else:
                splits += [s, e] if e != e_f else [s]
                take_inds.append(ind + 1)

            ind += 1 if s - e_p <= 1.0 else 2
            e_p = e

        segment_times = ",".join([str(spl) for spl in splits])

        with tempfile.TemporaryDirectory() as tmpdir:
            # TODO: we need to put the extension into the metadata
            # TODO: This can be done better using pipes I just don't feel like sinking too much time into this rn
            with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                f.write(video_bytes)
            try:
                _ = (
                    ffmpeg.input(f"{tmpdir}/input.mp4", ss=s_0, to=e_f)
                    .output(
                        f"{tmpdir}/clip_%d.mp4",
                        c="copy",
                        map=0,
                        f="segment",
                        segment_times=segment_times,
                        reset_timestamps=1,
                    )
                    .run(capture_stdout=True, quiet=True)
                )
            except Exception as err:  # pylint: disable=broad-except
                return [], [], str(err)

            video_clips = glob.glob(f"{tmpdir}/clip*")
            correct_clips = []
            for clip_id, (clip, ind) in enumerate(zip(clips, take_inds)):
                if ind < len(video_clips):
                    correct_clips.append((clip_id, clip, video_clips[ind]))
            # clips_lost = len(take_inds) - len(correct_clips) # TODO report this somehow

            video_clips, metadata_clips = [], []
            for clip_id, clip_span, clip_pth in correct_clips:
                with open(clip_pth, "rb") as vid_f:
                    clip_bytes = vid_f.read()
                video_clips.append(clip_bytes)

                clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                    clip_id=clip_id, oom_clip_count=self.oom_clip_count
                )
                meta_clip = metadata.copy()
                # set the timeframe of this clip
                meta_clip["clips"] = [clip_span]
                meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"
                metadata_clips.append(meta_clip)

        # TODO: subtitle chopping
        return video_clips, metadata_clips, None
