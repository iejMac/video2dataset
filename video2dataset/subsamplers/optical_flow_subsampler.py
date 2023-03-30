"""
optical flow detection
"""
import cv2
import numpy as np

try:
    from raft import RAFT
    from raft.utils.utils import InputPadder
    import torch
except:  # pylint: disable=broad-except,bare-except
    pass


class AttrDict(dict):
    """
    Lets us access dict keys with <dict>.key
    """

    # pylint: disable=super-with-arguments
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def resize_image_with_aspect_ratio(image, target_shortest_side=16):
    """
    Resize an input image while maintaining its aspect ratio.

    This function takes an image and resizes it so that its shortest side
    matches the specified target length. The other side is scaled accordingly
    to maintain the original aspect ratio of the image. The function returns
    the resized image and the scaling factor used for resizing.

    Parameters
    ----------
    image : numpy.ndarray
        The input image represented as a NumPy array with shape (height, width, channels).
    target_shortest_side : int, optional
        The desired length for the shortest side of the resized image (default is 16).

    Returns
    -------
    resized_image : numpy.ndarray
        The resized image with the same number of channels as the input image.
    scaling_factor : float
        The scaling factor used to resize the image.
    """
    # Get the original dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new dimensions while maintaining the aspect ratio
    if height < width:
        new_height = target_shortest_side
        new_width = int(width * (target_shortest_side / height))
        scaling_factor = height / new_height
    else:
        new_width = target_shortest_side
        new_height = int(height * (target_shortest_side / width))
        scaling_factor = width / new_width
    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image, scaling_factor


class RAFTDetector:
    """
    Optical flow detection using RAFT. Steps to setup RAFT:

    1. clone this repo fork: https://github.com/danielmend/RAFT
    2. in the root directory of the repo, run `pip install -e .` to install RAFT locally
    3. download optical flow models using the `download.sh` script in the repo, i.e. run `bash download.sh`
    4. when passing in args to the video2dataset call, be sure to include optical flow params as follows:
    args = {
        'model': '/path/to/optical/flow.pth',
        'path': None,
        'small': False,
        'mixed_precision': True,
        'alternate_corr': False,
    }
    """

    def __init__(self, args, downsample_size=None):

        self.device = args.get("device", "cuda")
        self.downsample_size = downsample_size

        model = RAFT(args)

        state_dict = torch.load(args.model)
        real_state_dict = {k.split("module.")[-1]: v for k, v in state_dict.items()}

        model.load_state_dict(real_state_dict)

        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess(self, frame1, frame2):
        """
        Preprocesses a pair of input frames for use with the RAFT optical flow detector.
        This function takes in two input frames, resizes them while maintaining the aspect
        ratio (if downsample_size is set), converts the frames to uint8 data type,
        creates torch tensors from the frames, permutes the axes, and pads the frames
        to the same size using the InputPadder.

        Args:
            frame1 (np.ndarray): The first input frame as a NumPy array with shape (height, width, 3).
            frame2 (np.ndarray): The second input frame as a NumPy array with shape (height, width, 3).

        Returns:
            tuple: A tuple containing three elements:
                - frame1 (torch.Tensor): The preprocessed first input frame as
                a torch tensor with shape (1, 3, height, width).

                - frame2 (torch.Tensor): The preprocessed second input frame as a
                torch tensor with shape (1, 3, height, width).

                - scaling_factor (float): The scaling factor used to resize the input
                frames while maintaining the aspect ratio. If no resizing is performed,
                the scaling factor is 1.
        """
        scaling_factor = 1
        if self.downsample_size:
            frame1, scaling_factor = resize_image_with_aspect_ratio(frame1, self.downsample_size)
            frame2, _ = resize_image_with_aspect_ratio(frame2, self.downsample_size)

        frame1 = frame1.astype(np.uint8)
        frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
        frame1 = frame1[None].to(self.device)

        frame2 = frame2.astype(np.uint8)
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float()
        frame2 = frame2[None].to(self.device)

        padder = InputPadder(frame1.shape)

        frame1, frame2 = padder.pad(frame1, frame2)
        return frame1, frame2, scaling_factor

    def __call__(self, frame1, frame2):
        frame1, frame2, scaling_factor = self.preprocess(frame1, frame2)
        with torch.no_grad():
            _, flow_up = self.model(frame1, frame2, iters=20, test_mode=True)
        return flow_up[0].permute(1, 2, 0).cpu().numpy() * scaling_factor


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

    def __init__(
        self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0, downsample_size=None
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.downsample_size = downsample_size

    def preprocess(self, frame):
        out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaling_factor = 1
        if self.downsample_size:
            out, scaling_factor = resize_image_with_aspect_ratio(out, self.downsample_size)

        return out, scaling_factor

    def __call__(self, frame1, frame2):
        """
        Calculate optical flow between two frames using Farneback method.

        Args:
            frame1 (numpy.ndarray): The first frame (grayscale).
            frame2 (numpy.ndarray): The second frame (grayscale).

        Returns:
            numpy.ndarray: The computed optical flow.
        """
        frame1, scaling_factor = self.preprocess(frame1)
        frame2, _ = self.preprocess(frame2)

        return (
            cv2.calcOpticalFlowFarneback(
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
            * scaling_factor
        )


class OpticalFlowSubsampler:
    """
    A class to detect optical flow in video frames.

    Attributes:
        detector (Cv2Detector or RAFTDetector): The optical flow detector.
        fps (int): The target frames per second. Defaults to -1 (original FPS).
    """

    def __init__(self, detector="cv2", fps=-1, args=None, downsample_size=None, dtype=np.float16):
        if detector == "cv2":
            if args:
                pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = args
                self.detector = Cv2Detector(
                    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags, downsample_size=downsample_size
                )
            else:
                self.detector = Cv2Detector(downsample_size=downsample_size)
        elif detector == "raft":
            assert args is not None
            if not isinstance(args, AttrDict):
                args = AttrDict(args)
            self.detector = RAFTDetector(args, downsample_size=downsample_size)
        else:
            raise NotImplementedError()

        self.fps = fps
        self.downsample_size = downsample_size
        self.dtype = dtype

    def __call__(self, frames, original_fps):
        optical_flow = []

        if self.fps == -1:
            self.fps = original_fps
            take_every_nth = 1
        elif self.fps > original_fps:
            take_every_nth = 1
        else:
            take_every_nth = int(round(original_fps / self.fps))

        try:
            frame1 = frames[0]
            prvs = frame1
            fc = 0

            for frame2 in frames[1:]:
                fc += 1
                if fc % take_every_nth != 0:
                    continue

                next_frame = frame2

                flow = self.detector(prvs, next_frame)

                optical_flow.append(flow)
                prvs = next_frame
        except Exception as err:  # pylint: disable=broad-except
            return [], None, str(err)

        opt_flow = np.array(optical_flow)
        mean_magnitude_per_frame = np.linalg.norm(opt_flow, axis=-1).mean(axis=(1, 2))
        mean_magnitude = float(mean_magnitude_per_frame.mean())

        metrics = [mean_magnitude, mean_magnitude_per_frame.tolist()]
        return opt_flow.astype(self.dtype), metrics, None
