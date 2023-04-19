"""Audio Decoders"""
import os
import re
from typing import Iterable
import torchaudio
import io
import subprocess as sp

torchaudio.set_audio_backend('soundfile')

def audio_to_wav(audio_bytes):

    cmd = [
        '/fsx/iejmac/ffmpeg2/ffmpeg',
        '-v', 'error',
        '-i', 'pipe:',
        '-f', 'wav',
        'pipe:'
        ]
    proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE, stdout=sp.PIPE)
    out, err = proc.communicate(audio_bytes)
    err = err.decode()
    proc.wait()
    return out


class AudioDecoder:
    def __init__(self, sample_rate=None, num_channels=None):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def __call__(self, key, data):
        extension = re.sub(r".*[.]", "", key)
        if extension not in "mp3 wav flac m4a".split():
            return None
        additional_info = dict()
        if extension != 'wav':
            wav_bytes = audio_to_wav(data)
        else:
            wav_bytes = data

        waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes), format='wav')
        additional_info['original_sample_rate'] = sample_rate
        return (waveform, additional_info)
  
