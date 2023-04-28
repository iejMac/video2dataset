"""Audio Decoders"""
import io
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as Fa


def set_backend(extension):
    """Sets torchaudio backend for different extensions (soundfile doesn't support M4A and MP3)"""
    if extension in ["wav", "flac"]:
        torchaudio.set_audio_backend("soundfile")
    else:
        torchaudio.set_audio_backend("sox_io")


class AudioDecoder:
    """Basic audio decoder that converts audio into torch tensors"""

    def __init__(self, sample_rate=48000, num_channels=None, extension="wav", max_length=10):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.max_length = max_length
        set_backend(extension)

    def __call__(self, key, data):
        extension = key.split(".")[-1]
        if extension not in "mp3 wav flac m4a".split():
            return None
        additional_info = {}
        waveform, sample_rate = torchaudio.load(io.BytesIO(data), format=extension)

        waveform = Fa.resample(waveform, sample_rate, self.sample_rate)
        pad_masks = torch.zeros((self.max_length * self.sample_rate,))
        pad_start = self.max_length * self.sample_rate - waveform.shape[1]

        if pad_start < 0:
            waveform = waveform[:, : self.max_length * self.sample_rate]
        if pad_start > 0:
            waveform = F.pad(waveform, (0, self.max_length * self.sample_rate - waveform.shape[1]), "constant")
            pad_masks[:pad_start] = 1.0

        additional_info["audio_pad_masks"] = pad_masks

        additional_info["original_sample_rate"] = sample_rate
        additional_info["sample_rate"] = self.sample_rate
        return (waveform, additional_info)
