"""Audio Decoders"""
import re
import torchaudio
import io


def set_backend(extension):
    """Sets torchaudio backend for different extensions (soundfile doesn't support M4A and MP3)"""
    if extension in ["wav", "flac"]:
        torchaudio.set_audio_backend("soundfile")
    else:
        torchaudio.set_audio_backend("sox_io")


class AudioDecoder:
    """Basic audio decoder that converts audio into torch tensors"""

    def __init__(self, sample_rate=None, num_channels=None, extension="wav"):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        set_backend(extension)

    def __call__(self, key, data):
        extension = key.split(".")[-1]
        if extension not in "mp3 wav flac m4a".split():
            return None
        additional_info = {}
        waveform, sample_rate = torchaudio.load(io.BytesIO(data), format=extension)
        additional_info["original_sample_rate"] = sample_rate
        return (waveform, additional_info)
