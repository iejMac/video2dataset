"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import ffmpeg
import tempfile


class ResolutionSubsampler:
    """
    Adjusts the resolution of the videos to the specified height and width

    TODO: for now will just implement keep_ratio + center_crop, in the future maybe add more options
    """

    def __init__(self, video_size):
        self.video_size = video_size

    def __call__(self, video_bytes):
        subsampled_bytes = []
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)
                try:
                    _ = (
                        ffmpeg.input(f"{tmpdir}/input.mp4")
                        .filter("scale", -2, self.video_size)
                        .filter("crop", w=self.video_size, h=self.video_size)
                        .filter("pad", w=self.video_size, h=self.video_size)
                        .output(f"{tmpdir}/output.mp4", reset_timestamps=1)
                        .run(capture_stdout=True, quiet=True)
                    )
                except Exception as err:  # pylint: disable=broad-except
                    return [], str(err)

                with open(f"{tmpdir}/output.mp4", "rb") as f:
                    subsampled_bytes.append(f.read())
        return subsampled_bytes, None
