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
        return [video_bytes.name], [metadata], None


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

    def __call__(self, video_file, metadata):
        clips = metadata.pop("clips")

        s_0, e_f = get_seconds(clips[0][0]), get_seconds(clips[-1][1])

        ind = 1
        _, e_p = clips[0]  # we assume there's always one clip which we want to take
        e_p = get_seconds(e_p)
        splits = [e_p]
        take_inds = [0]  # list of indicies of clips to take, used to discard non-contiguous sections

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

        tmpdir = tempfile.TemporaryDirectory()

        try:
            _ = (
                ffmpeg.input(str(video_file.name), ss=s_0, to=e_f)
                .output(
                    f"{tmpdir.name}/clip_%d.mp4",
                    c="copy",
                    map=0,
                    f="segment",
                    segment_times=segment_times,
                    reset_timestamps=1,
                )
                .run(capture_stdout=True, quiet=True)
            )
        except Exception as err:  # pylint: disable=broad-except
            print(err)
            return None, [], [], str(err)

        video_clips = glob.glob(f"{tmpdir.name}/clip*")
        correct_clips = []
        for clip_id, (clip, ind) in enumerate(zip(clips, take_inds)):
            if ind < len(video_clips):
                correct_clips.append((clip_id, clip, video_clips[ind]))
        # clips_lost = len(take_inds) - len(correct_clips) # TODO report this somehow

        video_clips, metadata_clips = [], []
        for clip_id, clip_span, clip_pth in correct_clips:
            # with open(clip_pth, "rb") as vid_f:
            #     clip_bytes = vid_f.read()
            video_clips.append(clip_pth)

            clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                clip_id=clip_id, oom_clip_count=self.oom_clip_count
            )
            meta_clip = metadata.copy()
            meta_clip["clips"] = [clip_span]  # set the timeframe of this clip
            meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"
            metadata_clips.append(meta_clip)

        # TODO: subtitle chopping
        return tmpdir, video_clips, metadata_clips, None
