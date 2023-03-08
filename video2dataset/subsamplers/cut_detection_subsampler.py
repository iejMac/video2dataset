from clipping_subsampler import ClippingSubsampler
import numpy as np
from scenedetect import detect, AdaptiveDetector


class CutDetectionSubsampler:
    def __init__(self, video_fps, cut_mode="all"):
        if cut_mode not in ["all", "longest"]:
            raise NotImplementedError()

        self.video_fps = video_fps
        self.cut_mode = cut_mode

    def __call__(self, streams):
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
        scene = scene / self.video_fps

        if self.cut_mode == "longest":
            longest_clip = np.argmax(
                [clip[1] - clip[0] for scene in clip]
            )
            return scene[longest_clip]
        else:
            return scene