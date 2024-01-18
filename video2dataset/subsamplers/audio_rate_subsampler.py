"""
frame subsampler adjusts the fps of the videos to some constant value
"""


import tempfile
import os
import ffmpeg


class AudioRateSubsampler:
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Args:
        sample_rate (int): Target sample rate of the audio.
        encode_format (str): Format to encode in (i.e. m4a)
    """

    def __init__(self, sample_rate, encode_format, n_audio_channels=None):
        self.sample_rate = sample_rate
        self.encode_format = encode_format
        self.n_audio_channels = n_audio_channels

    def __call__(self, streams, metadata=None):
        audio_bytes = streams.pop("audio")
        subsampled_bytes = []
        for aud_bytes in audio_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.m4a"), "wb") as f:
                    f.write(aud_bytes)
                ext = self.encode_format
                try:
                    # TODO: for now assuming m4a, change this
                    ffmpeg_args = {"ar": str(self.sample_rate), "f": ext}
                    if self.n_audio_channels is not None:
                        ffmpeg_args["ac"] = str(self.n_audio_channels)
                    _ = ffmpeg.input(f"{tmpdir}/input.m4a")
                    _ = _.output(f"{tmpdir}/output.{ext}", **ffmpeg_args).run(capture_stdout=True, quiet=True)
                except Exception as err:  # pylint: disable=broad-except
                    return [], None, str(err)

                with open(f"{tmpdir}/output.{ext}", "rb") as f:
                    subsampled_bytes.append(f.read())
        streams["audio"] = subsampled_bytes
        return streams, metadata, None
