import numpy as np
from scenedetect import detect, AdaptiveDetector
import os
import tempfile
import cv2
class CutDetectionSubsampler:
    def __init__(self, cut_detection_mode="all"):
        if cut_detection_mode not in ["all", "longest"]:
            raise NotImplementedError()

        self.cut_detection_mode = cut_detection_mode

    def __call__(self, streams):
        video_bytes = streams['video']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "input.mp4"), "wb") as f:
                f.write(video_bytes)

            scene_list = detect(os.path.join(tmpdir, "input.mp4"), AdaptiveDetector(), start_in_scene=True)
            scene = []
            for clip in scene_list:
                scene.append((clip[0].get_frames(), clip[1].get_frames()))
            video_fps = cv2.VideoCapture(os.path.join(tmpdir, "input.mp4")).get(cv2.CAP_PROP_FPS)
        scene = np.array(scene)
        scene = (scene / video_fps).tolist()
        
        if self.cut_detection_mode == "longest":
            longest_clip = np.argmax(
                [clip[1] - clip[0] for clip in scene]
            )
            return scene[longest_clip]
        else:
            return scene