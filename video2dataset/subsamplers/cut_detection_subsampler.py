from clipping_subsampler import ClippingSubsampler
import numpy as np
from scenedetect import detect, AdaptiveDetector


class CutDetectionSubsampler:
    def __init__(self, oom_clip_count, encode_formats):
        self.oom_clip_count = oom_clip_count
        self.encode_formats = encode_formats

        self.clip_subsampler = ClippingSubsampler(oom_clip_count, encode_formats)

    def __call__(self, streams, metadata):
        video_bytes = streams['video']
        for vid_bytes in video_bytes:
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                    f.write(vid_bytes)

        scene_list = detect("input.mp4", AdaptiveDetector(), start_in_scene=True)
        scene = []
        for clip in scene_list:
            scene.append((clip[0].get_frames(), clip[1].get_frames()))
        scene = np.array(scene)

        metadata['clips'] = scene
        stream_fragments, meta_fragments, error_message = self.clip_subsampler(streams, metadata)

        return stream_fragments, meta_fragments, error_message