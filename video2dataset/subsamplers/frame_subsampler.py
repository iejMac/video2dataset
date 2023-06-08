"""
frame subsampler adjusts the fps of the videos to some constant value
"""


import tempfile
import os
import ffmpeg

from .subsampler import Subsampler


class FrameSubsampler(Subsampler):
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Subsamples the frames of the video in terms of spacing and quantity (frame_rate, which ones etc.)
    Args:
        frame_rate (int): Target frame rate of the videos.
        downsample_method (str): determiens how to downsample the video frames:
            fps: decreases the framerate but sample remains a valid video
            first_frame: only use the first frame of a video of a video and output as image

    TODO: n_frame
    TODO: generalize interface, should be like (frame_rate, n_frames, sampler, output_format)
    # frame_rate - spacing
    # n_frames - quantity
    # sampler - from start, end, center out
    # output_format - save as video, or images
    """

    def __init__(self, frame_rate, downsample_method="fps"):
        self.frame_rate = frame_rate
        self.downsample_method = downsample_method
        self.output_modality = "video" if downsample_method == "fps" else "jpg"

    def __call__(self, streams, metadata=None):
        # video_bytes = streams["video"]
        # TODO": this should be an arg (replace or add)
        video_bytes = streams.pop("video")
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    ext = "mp4"
                    if self.downsample_method == "fps":
                        _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                        _ = _.filter("fps", fps=self.frame_rate)
                        _ = _.output(f"{tmpdir}/output.mp4", reset_timestamps=1).run(capture_stdout=True, quiet=True)
                    elif "frame" in self.downsample_method:
                        _ = ffmpeg.input(f"{tmpdir}/input.mp4")
                        _ = _.filter("select", f"eq(n,0)")
                        _ = _.output(f"{tmpdir}/output.jpg").run(capture_stdout=True, quiet=True)
                        ext  = "jpg"

                except Exception as err:  # pylint: disable=broad-except
                    return [], None, str(err)

                with open(f"{tmpdir}/output.{ext}", "rb") as f:
                    subsampled_bytes.append(f.read())
        streams[self.output_modality] = subsampled_bytes
        return streams, None, None
