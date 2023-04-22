"""Audio Decoders"""
import torchaudio
import io
import torch

torchaudio.set_audio_backend("sox_io")



class AudioDecoder:
    """Basic audio decoder that converts audio into torch tensors"""

    def __init__(self, sample_rate=None, num_channels=None):
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    def __call__(self, key, data):
        extension = key.split('.')[-1]

        if extension not in "mp3 wav flac m4a".split():
            return None
        additional_info = {}
        wav_bytes = data
        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes), format=extension)
        additional_info["original_sample_rate"] = sample_rate
        return (waveform, additional_info)
