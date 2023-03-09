import numpy as np
from scenedetect import AdaptiveDetector, SceneManager, SceneDetector, open_video
import os
import tempfile
import cv2

def get_scenes_from_scene_manager(scene_manager, cut_detection_mode):
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    scene = []
    
    for clip in scene_list:
        scene.append([clip[0].get_frames(), clip[1].get_frames()])

    if cut_detection_mode == "longest": # we have multiple cuts, pick the longest
        longest_clip = np.argmax(
            [clip[1] - clip[0] for clip in scene]
        )
        scene = [scene[longest_clip]]

    return scene

class CutDetectionSubsampler:
    def __init__(self, cut_detection_mode="all", framerates=[]):
        if cut_detection_mode not in ["all", "longest"]:
            raise NotImplementedError()
        self.framerates = framerates
        self.cut_detection_mode = cut_detection_mode

    def __call__(self, streams):
        video_bytes = streams['video']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            video = open_video(video_path)
        
            detector = AdaptiveDetector()
            scene_manager = SceneManager()
            scene_manager.add_detector(detector)

            cuts = {}
            original_fps = video.frame_rate
            cuts["original_fps"] = original_fps
            
            scene_manager.detect_scenes(video=video)
            cuts["cuts_original_fps"] = get_scenes_from_scene_manager(scene_manager, self.cut_detection_mode)

            for target_fps in self.framerates:
                video.reset()
                
                scene_manager = SceneManager()
                detector = AdaptiveDetector()
                scene_manager.add_detector(detector)
                frame_skip = max(int(original_fps//target_fps) - 1, 0) # keep in mind this is skipping frames, so we need to subtract one from the target sampling ratio
                
                scene_manager.detect_scenes(video=video, frame_skip=fps_ratio)
                cuts[f"cuts_{target_fps}"] = get_scenes_from_scene_manager(scene_manager, self.cut_detection_mode)
                scene_manager.clear()
                
        print(cuts)
        return cuts