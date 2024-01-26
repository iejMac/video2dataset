"""
cut detection subsampler detects cuts in a video
"""
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video
from typing import Tuple

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, TempFilepaths

# TODO: this can be done more elegantly:
# from scenedetect import scene_manager and set that in correct namespace
# best solution is just figure out best value for them and submit PR
DEFAULT_MIN_WIDTH = 64


def get_scenes_from_scene_manager(scene_manager, cut_detection_mode):
    """
    Returns a list of cuts from a scene manager given a cut detection mode
    """
    scene_list = scene_manager.get_scene_list(start_in_scene=True)
    scene = []

    for clip in scene_list:
        scene.append([clip[0].get_frames(), clip[1].get_frames()])

    if cut_detection_mode == "longest":  # we have multiple cuts, pick the longest
        longest_clip = np.argmax([clip[1] - clip[0] for clip in scene])
        scene = [scene[longest_clip]]

    return scene


class CutDetectionSubsampler(Subsampler):
    """
    Detects cuts in input videos and returns contiguous segments in a video as metadata.

    non-initialization args:
    - cuts_are_clips: whether to create video clips from the source video based on cuts

    initialization args:
    - cut_detection_mode to be either "longest" to pick the longest cut or "all" to pick all cuts
    - framerates to be None (for original fps only) or a list of target framerates to detect cuts in
    - threshold - determines roughly how much motion is required for a "cut" (tunable parameter)
    - min_scene_len - minimum scene length to not drop a scene (see pyscenedeteect docs for more explanation)
    """

    def __init__(self, cut_detection_mode="all", framerates=None, threshold=27, min_scene_len=15):
        self.framerates = framerates
        self.cut_detection_mode = cut_detection_mode
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def __call__(self, filepaths: TempFilepaths, metadata: Metadata) -> Tuple[TempFilepaths, Metadata, Error]:
        try:
            # process first video
            assert "video" in filepaths
            filepath = filepaths["video"][0]
            video = open_video(filepath)

            detector = ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
            scene_manager = SceneManager()
            scene_manager.add_detector(detector)
            scene_manager.auto_downscale = False
            scene_manager.downscale = video.frame_size[0] // DEFAULT_MIN_WIDTH

            cuts = {}
            original_fps = video.frame_rate
            cuts["original_fps"] = original_fps

            scene_manager.detect_scenes(video=video)
            cuts["cuts_original_fps"] = get_scenes_from_scene_manager(scene_manager, self.cut_detection_mode)
            if self.framerates is not None:
                for target_fps in self.framerates:
                    video.reset()

                    detector = ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len)
                    scene_manager = SceneManager()
                    scene_manager.add_detector(detector)
                    frame_skip = max(
                        int(original_fps // target_fps) - 1, 0
                    )  # if we take 1 frame and skip N frames we're sampling 1/N+1 % of the video
                    # so if we desire to sample 1/N of the video, we need to subtract one when doing frame skipping

                    scene_manager.detect_scenes(video=video, frame_skip=frame_skip)
                    cuts[f"cuts_{target_fps}"] = get_scenes_from_scene_manager(
                        scene_manager, self.cut_detection_mode
                    )
                    scene_manager.clear()
            metadata["cuts"] = cuts
        except Exception as err:  # pylint: disable=broad-except
            return filepaths, metadata, str(err)

        return filepaths, metadata, None
