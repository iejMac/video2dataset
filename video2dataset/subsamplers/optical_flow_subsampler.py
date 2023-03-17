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

class OpticalFlowSubsampler:
    """
    Detects optical flow in video frames
    """

    def __init__(self, detector="cv2", take_every_nth=1):
        self.take_every_nth = take_every_nth
        self.detector = detector

    def __call__(self, video_bytes):
        optical_flow = []
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'input.mp4')
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            ret, frame1 = cap.read()
            prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            fc = 0
            
            while ret:
                ret, frame2 = cap.read()
                fc += 1

                if fc % self.take_every_nth != 0:
                    continue
                
                next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                optical_flow.append(flow)
                prvs = next

        return optical_flow
