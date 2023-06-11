"""
WhisperX subsampler - transcribes audio using the Whisper model from OAI

code: https://github.com/m-bain/whisperX
"""
import os
import tempfile

try:
    import whisperx
    import torch
except:  # pylint: disable=broad-except,bare-except
    pass

from .subsampler import Subsampler


class WhisperSubsampler(Subsampler):
    """
    Transcribes audio samples using the OAI Whisper Model via WhisperX API

    Params:
        model_name: https://github.com/guillaumekln/faster-whisper/blob/20d4e9418b5efb69ec5aa4819a39e3fb0e772a2a/faster_whisper/transcribe.py#LL90C1-L90C1
        batch_size: batch size used during inference (try to maximize this for perf)
        compute_type: accuracy/mem tradeoff (float16, float32, int8)
    """

    def __init__(
        self,
        model_name="large-v2",
        batch_size=16,
        compute_type="float16",
        is_slurm_task=False,
    ):
        if is_slurm_task:
            local_rank = os.environ["LOCAL_RANK"]
            device = f"cuda:{local_rank}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisperx.load_model(model_name, device, compute_type=compute_type)
        self.batch_size = batch_size

    def __call__(self, streams, metadata=None):
        audio_bytes = streams.get("audio")

        for i, aud_bytes in enumerate(audio_bytes):
            # TODO: .m4a not always
            with tempfile.NamedTemporaryFile(suffix=".m4a") as tmpfile:
                tmpfile.write(aud_bytes)
                tmpfile.flush()  # ensure all data is written
                try:
                    audio = whisperx.load_audio(tmpfile.name)
                    result = self.model.transcribe(audio, batch_size=self.batch_size)
                    metadata[i]["whisper_transcript"] = result
                except Exception as err:  # pylint: disable=broad-except
                    return [], {}, str(err)

        return streams, metadata, None
