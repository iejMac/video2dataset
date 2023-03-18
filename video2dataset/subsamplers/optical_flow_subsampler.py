"""
optical flow detection
"""
import os
import copy
import glob
import ffmpeg
import tempfile
import cv2
from datetime import datetime
import torch
import numpy as np

class Cv2Detector:
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def __call__(self, frame1, frame2):
        return cv2.calcOpticalFlowFarneback(frame1, frame2, None, self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n, self.poly_sigma, self.flags)
        
class OpticalFlowSubsampler:
    """
    Detects optical flow in video frames
    """

    def __init__(self, detector="cv2", fps=-1, cv2_params=None, raft_params=None):
        if detector == "cv2":
            if cv2_params:
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = cv2_params
                self.detector = Cv2Detector(pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            else:
                self.detector = Cv2Detector()
        self.fps = fps

    def __call__(self, video_bytes):
        optical_flow = []
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'input.mp4')
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            try:
                cap = cv2.VideoCapture(video_path)
                original_fps = cap.get(cv2.CAP_PROP_FPS)

                if self.fps == -1:
                    self.fps = original_fps
                    take_every_nth = 1
                elif self.fps > original_fps:
                    take_every_nth = 1
                else:
                    take_every_nth = int(round(original_fps / self.fps))

                ret, frame1 = cap.read()
                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                fc = 0
                
                while True:
                    ret, frame2 = cap.read()
                    fc += 1

                    if not ret:
                        break

                    if fc % take_every_nth != 0:
                        continue

                    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    flow = self.detector(prvs, next)

                    optical_flow.append(flow)
                    prvs = next

            except Exception as err:  # pylint: disable=broad-except
                return [], str(err)

        return np.array(optical_flow, dtype=np.float16), None
