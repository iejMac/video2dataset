"""
detects optical flow in a video
"""
import numpy as np
import os
import tempfile
import cv2

class OpticalFlowSubsampler:
    """
    Detects optical flow in input videos and returns the optical flow metadata.

    expects:
    - feature_params to be either "longest" to pick the longest cut or "all" to pick all cuts
    - framerates to be None (for original fps only) or a list of target framerates to detect cuts in
    """

    def __init__(self, feature_params=None, lk_params=None):
        self.feature_params = feature_params or dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        ) # default params
        
        self.lk_params = lk_params or dict(
            winSize=(15,15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        ) # default params

    def __call__(self, streams):
        video_bytes = streams["video"]

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(video_path)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fc = 1
            ret, frame = cap.read()
            
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **self.feature_params)
            
            # TODO: find a better way to handle when this happens
            if p0 is None: # cv2.goodFeaturesToTrack returns no good features so we cant track optical flow
                return None

            metric = 0.

            while (fc < frameCount and ret):
                
                ret, frame = cap.read()
                fc += 1

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
                p1 = p1.astype(np.float32)
                # Select good points
                good_new = p1  # [st==1]
                good_old = p0  # [st==1]

                # Compute metric
                metric += np.linalg.norm(good_new - good_old, axis=1)

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)
                p0 = p0.astype(np.float32)
                
            return metric.tolist()