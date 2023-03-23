"""
Worker for optical flow stage
"""
import time
import pyarrow as pa
import traceback
import io
import numpy as np
import fsspec
import tempfile
import os
import cv2

from video2dataset.logger import CappedCounter, write_stats
from video2dataset.subsamplers import OpticalFlowSubsampler
from video2dataset.dataloader import get_video_dataset


def numpy_npz_dumps(numpy_dict):
    """
    Dump a dictionary of numpy arrays into a bytestring using numpy npz format.

    Args:
        numpy_dict (dict): A dictionary containing numpy arrays as values.

    Returns:
        bytes: A bytestring representing the compressed numpy arrays.
    """

    stream = io.BytesIO()
    np.savez_compressed(stream, **numpy_dict)
    return stream.getvalue()


def frames_to_mp4_bytes(frames, fps=30, codec="mp4v"):
    """
    Convert a list of frames to an mp4 video encoded as bytes.

    Args:
        frames (list): A list of frames.
        fps (int): The frames per second of the video. Default is 30.
        codec (str): The codec to use for encoding. Default is 'mp4v'.

    Returns:
        bytes: A bytestring representing the encoded video.
    """
    # Get frame dimensions from the first frame
    height, width, _ = frames[0].shape

    # Create a temporary file to save the video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write the frames to the video
        for frame in frames:
            out.write(frame)

        # Release the VideoWriter
        out.release()

        # Read the video file as bytes
        with open(video_path, "rb") as f:
            video_bytes = f.read()

    # Remove the temporary video file
    os.remove(video_path)

    return video_bytes


def convert_frames_depth(frames, target_depth=np.uint8):
    """
    Convert the frame depth of a list of frames.

    Args:
        frames (list): A list of frames.
        target_depth (numpy.dtype): The target depth. Default is np.uint8.

    Returns:
        list: A list of converted frames.
    """
    converted_frames = []
    for frame in frames:
        converted_frame = frame.astype(target_depth)
        converted_frames.append(converted_frame)
    return converted_frames


class OpticalFlowWorker:
    """
    A class to read shards, process them using OpticalFlowSubsampler, and write the output.

    Attributes:
        sample_writer_class (type): The class used to write samples.
        output_folder (str): The folder to write the output.
        thread_count (int): The number of threads.
        number_sample_per_shard (int): The number of samples per shard.
        oom_shard_count (int): The number of out-of-memory shards.
        encode_formats (dict): The encoding formats.
        detector (str): The optical flow detector type.
        fps (int): The target frames per second.
        optical_flow_subsampler (OpticalFlowSubsampler): The OpticalFlowSubsampler instance.
    """

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        thread_count,
        number_sample_per_shard,
        oom_shard_count,
        encode_formats,
        detector,
        fps,
        downsample_dims,
        dtype
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.encode_formats = encode_formats
        self.save_caption = True
        self.detector = detector
        self.fps = fps
        self.downsample_dims = downsample_dims
        self.dtype = dtype

        self.optical_flow_subsampler = OpticalFlowSubsampler(
            detector=detector, 
            fps=fps,
            downsample_dims = downsample_dims,
            dtype=dtype
        )

    def __call__(
        self,
        row,
    ):
        try:
            self.process_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def process_shard(
        self,
        row,
    ):
        """
        Process a video shard using the OpticalFlowSubsampler.

        Args:
            row (tuple): A tuple containing the shard and shard_id.

        Raises:
            Except
        """
        shard, shard_id = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")
        with fs.open(shard_path, "rb") as f:
            df = pa.parquet.read_table(f)
            schema = df.schema

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )

        successes = 0
        failed_to_subsample = 0

        decoder_kwargs = {"n_frames": None, "fps": None, "num_threads": 4, "return_bytes": True}

        dset = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs=decoder_kwargs,
            resize_size=None,
            crop_size=None,
        )

        for sample in dset:
            key = sample["__key__"][0]
            caption = sample.get("txt", b"")[0]
            meta = sample.get("json", {})[0]

            streams = {}
            frames = np.array(sample.get("mp4")[0])
            native_fps = sample.get("native_fps").item()

            for mod, fmt in self.encode_formats.items():
                streams[mod] = sample.get(fmt, b"")

            optical_flow, metrics, error_message = self.optical_flow_subsampler(frames, native_fps)

            if error_message is not None:
                failed_to_subsample += 1
                status = "failed_to_subsample"
                status_dict.increment(error_message)
                meta["status"] = status
                meta["error_message"] = error_message
                sample_writer.write(
                    {},
                    key,
                    caption,
                    meta,
                )
                continue

            successes += 1
            status = "success"
            status_dict.increment(status)
            meta["status"] = status

            mean_magnitude, mean_magnitude_per_frame = metrics
            meta["mean_optical_flow_magnitude"] = mean_magnitude
            meta["mean_optical_flow_magnitude_per_frame"] = mean_magnitude_per_frame
            meta["optical_flow_fps"] = self.fps
            if self.downsample_dims:
                meta["optical_flow_downsample_dims"] = self.downsample_dims
            meta["optical_flow_dtype"] = str(self.dtype)

            streams["numpy_metadata"] = sample.get("npz", {})
            if isinstance(streams["numpy_metadata"], bytes):
                npz_bytes = io.BytesIO(streams["numpy_metadata"])
                streams["numpy_metadata"] = dict(np.load(npz_bytes))
            streams["numpy_metadata"]["optical_flow"] = optical_flow
            streams["numpy_metadata"] = numpy_npz_dumps(streams["numpy_metadata"])

            #input_frames = convert_frames_depth(frames[:, :, :, ::-1], target_depth=np.uint8)
            #mp4_bytes = frames_to_mp4_bytes(input_frames, fps=native_fps)
            streams["video"] = sample.get("video_bytes")[0]

            sample_writer.write(
                streams,
                key,
                caption,
                meta,
            )

        sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            1,  # count
            successes,
            0,  # failed to download
            failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
