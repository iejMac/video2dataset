"""
optical flow detection
"""
import os
import tempfile
import cv2
import numpy as np
import torch

from raft.raft import RAFT
from raft.utils import InputPadder

class Cv2Detector:
    """
    A class to perform optical flow detection using OpenCV's Farneback method.

    Attributes:
        pyr_scale (float): The pyramid scale. Defaults to 0.5.
        levels (int): The number of pyramid layers. Defaults to 3.
        winsize (int): The window size. Defaults to 15.
        iterations (int): The number of iterations. Defaults to 3.
        poly_n (int): The size of the pixel neighborhood. Defaults to 5.
        poly_sigma (float): The standard deviation of the Gaussian. Defaults to 1.2.
        flags (int): Additional flags for the cv2.calcOpticalFlowFarneback function. Defaults to 0.
    """

    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def preprocess(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def __call__(self, frame1, frame2):
        """
        Calculate optical flow between two frames using Farneback method.

        Args:
            frame1 (numpy.ndarray): The first frame (grayscale).
            frame2 (numpy.ndarray): The second frame (grayscale).

        Returns:
            numpy.ndarray: The computed optical flow.
        """
        frame1, frame2 = self.preprocess(frame1), self.preprocess(frame2)
        return cv2.calcOpticalFlowFarneback(
            frame1,
            frame2,
            None,
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            self.flags,
        )


class OpticalFlowSubsampler:
    """
    A class to detect optical flow in video frames.

    Attributes:
        detector (Cv2Detector or RAFTDetector): The optical flow detector.
        fps (int): The target frames per second. Defaults to -1 (original FPS).
    """

    def __init__(self, detector="cv2", fps=-1, params=None, device=None):
        if detector == "cv2":
            if params:
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = params
                self.detector = Cv2Detector(pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            else:
                self.detector = Cv2Detector()
        else:
            raise NotImplementedError()

        self.fps = fps

    def __call__(self, video_bytes):
        optical_flow = []
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
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
                prvs = frame1
                fc = 0

                while True:
                    ret, frame2 = cap.read()
                    fc += 1

                    if not ret:
                        break

                    if fc % take_every_nth != 0:
                        continue

                    next_frame = frame2

                    flow = self.detector(prvs, next_frame)

                    optical_flow.append(flow)
                    prvs = next_frame

            except Exception as err:  # pylint: disable=broad-except
                return [], str(err)

        return np.array(optical_flow, dtype=np.float16), None
