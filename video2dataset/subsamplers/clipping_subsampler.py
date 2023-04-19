"""
clipping subsampler turns full videos into clips of videos according to clip_col

TODO: implement subtitle splitting (can be done just by indexing subtitle dict during clipping
"""
import os
import copy
import glob
import ffmpeg
import tempfile

from datetime import datetime


def get_seconds(t):
    if not isinstance(t, str):
        return float(t)  # already seconds
    time_format = "%H:%M:%S.%f"  # TODO: maybe paramaterize this?
    t_obj = datetime.strptime(t, time_format).time()
    return t_obj.second + t_obj.microsecond / 1e6 + t_obj.minute * 60 + t_obj.hour * 3600


def split_time_frame(s, e, max_length):
    time_format = "%H:%M:%S.%f"  # TODO: maybe paramaterize this?
    start_time, end_time = datetime.strptime(s, time_format), datetime.strptime(e, time_format)
    time_difference = (end_time - start_time).total_seconds()

    time_frames = [
        (
            (start_time + timedelta(seconds=i * max_length)).strftime(time_format),
            (start_time + timedelta(seconds=min((i + 1) * max_length, time_difference))).strftime(time_format),
        )
        for i in range(int(time_difference // max_length) + (1 if time_difference % max_length > 0 else 0))
    ]
    return time_frames


class ClippingSubsampler:
    """
    Cuts videos up into segments according to the 'clips' metadata

    Parameters:
    oom_clip_count: int
        The number of orders of magnitude for clip count, used for formatting clip keys.
    encode_formats: dict
        A dictionary mapping stream keys to their corresponding file extensions, e.g., {"video": "mp4", "audio": "mp3"}.
    min_length: float optional (default=0.0)
        Minimum length in seconds of a clip. Below this the subsampler will reject the clips
    max_length: float optional (default=999999.0)
        Maximum clip length, if exceeded resolve according to max_length_strategy
    max_length_strategy: str optional (defaul="all")
        "all" - cut up long clip into as many clips of max_length as possible
        "first" - take the first max_length clip from the long clip
    precise: bool, optional (default=False)
        If True, provides more precise clipping at the expense of processing speed.
        If False, prioritizes speed over precision.

    expects:
    - clips to be sorted in increasing order and non-overlapping
    - time to be in the format "%H:%M:%S.%f", or a number representing the second of the timestamp
    """

    def __init__(
        self,
        oom_clip_count,
        encode_formats,
        min_length=0.0,
        max_length=999999.0,
        max_length_strategy="all",
        precise=False,
    ):
        self.oom_clip_count = oom_clip_count
        self.encode_formats = encode_formats
        self.min_length = min_length
        self.max_length, self.max_length_strategy = max_length, max_length_strategy
        self.precise = precise

    def __call__(self, streams, metadata):
        clips = metadata.pop("clips")

        if isinstance(clips[0], float):  # make sure clips looks like [[start, end]] and not [start, end]
            clips = [clips]

        filtered_clips = []
        for s, e in clips:
            max_len_clips = split_time_frame(s, e, self.max_length)
            last_time_d = get_seconds(max_len_clips[-1][1]) - get_seconds(max_len_clips[-1][0])
            max_len_clips = max_len_clips if last_time_d >= self.min_length else max_len_clips[:-1]

            if max_length_strategy == "first":
                max_len_clips = max_len_clips[:1]

            filtered_clips += max_len_clips
        clips = filtered_clips

        if len(clips) == 0:
            # return an error
            return {}, [], f"Video had no clips longer than {self.min_length}"

        start_0 = get_seconds(clips[0][0]) == 0.0

        ind = 1 + int(not start_0)
        s_p, e_p = clips[0]
        s_p, e_p = get_seconds(s_p), get_seconds(e_p)
        splits = (not start_0) * [s_p] + [e_p]
        # list of indicies of clips to take, used to discard non-contiguous sections
        take_inds = [int(not start_0)]

        # TODO: make nicer
        for s, e in clips[1:]:
            s, e = get_seconds(s), get_seconds(e)

            if s == e_p:  # situations like [0, 1], [1, 2], [2, 3] -> 1, 2
                splits += [e]
                take_inds.append(ind)
                ind += 1
            else:
                splits += [s, e]
                take_inds.append(ind + 1)
                ind += 2
            e_p = e

        segment_times = ",".join([str(spl) for spl in splits])
        streams_clips = {}

        for k in streams.keys():
            stream_bytes = streams[k]
            if stream_bytes is None:
                continue
            encode_format = self.encode_formats[k]

            with tempfile.TemporaryDirectory() as tmpdir:
                # TODO: we need to put the extension into the metadata
                # TODO: This can be done better using pipes I just don't feel like sinking too much time into this rn
                with open(os.path.join(tmpdir, f"input.{encode_format}"), "wb") as f:
                    f.write(stream_bytes)
                try:
                    kwargs = {
                        "map": 0,
                        "f": "segment",
                        "segment_times": segment_times,
                        "reset_timestamps": 1,
                    }

                    # Precision things, tradeoff for speed
                    if not self.precise:
                        kwargs["c"] = "copy"
                    else:
                        kwargs["force_key_frames"] = segment_times

                    _ = (
                        ffmpeg.input(f"{tmpdir}/input.{encode_format}")
                        .output(f"{tmpdir}/clip_%d.{encode_format}", **kwargs)
                        .run(capture_stdout=True, quiet=True)
                    )

                except Exception as err:  # pylint: disable=broad-except
                    return {}, [], str(err)

                stream_clips = glob.glob(f"{tmpdir}/clip*.{encode_format}")
                stream_clips.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

                correct_clips = []
                for clip_id, (clip, ind) in enumerate(zip(clips, take_inds)):
                    if ind < len(stream_clips):
                        correct_clips.append((clip_id, clip, stream_clips[ind]))
                # clips_lost = len(take_inds) - len(correct_clips) # TODO report this somehow

                stream_clips, metadata_clips = [], []
                for clip_id, clip_span, clip_pth in correct_clips:
                    with open(clip_pth, "rb") as vid_f:
                        clip_bytes = vid_f.read()
                    stream_clips.append(clip_bytes)

                    clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
                        clip_id=clip_id, oom_clip_count=self.oom_clip_count
                    )
                    meta_clip = copy.deepcopy(metadata)
                    # set the timeframe of this clip
                    meta_clip["clips"] = [clip_span]
                    meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"

                    if "subtitles" in meta_clip.get("yt_meta_dict", {}):
                        clip_subtitles = []
                        s_c, e_c = get_seconds(clip_span[0]), get_seconds(clip_span[1])
                        for line in meta_clip["yt_meta_dict"]["subtitles"]:
                            s, e = get_seconds(line["start"]), get_seconds(line["end"])
                            if max(s_c, s) < min(e_c, e):
                                clip_subtitles.append(line)
                            elif s > e_c:
                                break
                        # full video subtitles might still be useful for context
                        meta_clip["clip_subtitles"] = clip_subtitles

                    metadata_clips.append(meta_clip)

                streams_clips[k] = stream_clips

        return streams_clips, metadata_clips, None
