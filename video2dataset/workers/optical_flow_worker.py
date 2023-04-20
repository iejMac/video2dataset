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
        optical_flow_params,
        batched,
        subsampler_batch_size,
        n_frames,
        is_slurm_task,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.encode_formats = encode_formats
        self.save_caption = True
        self.detector = optical_flow_params.get("detector", "cv2")
        self.fps = optical_flow_params.get("fps", -1)
        self.downsample_size = optical_flow_params.get("downsample_size", None)
        self.dtype = optical_flow_params.get("dtype", "fp16")
        self.detector_args = optical_flow_params.get("detector_args", None)
        self.batched = batched
        self.subsampler_batch_size = subsampler_batch_size
        self.n_frames = n_frames

        self.optical_flow_subsampler = OpticalFlowSubsampler(
            detector=self.detector,
            args=self.detector_args,
            fps=self.fps,
            downsample_size=self.downsample_size,
            dtype=self.dtype,
            is_slurm_task=is_slurm_task,
            batched=batched,
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
        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")

            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception as e: # pylint: disable=broad-except
            fields = [
                pa.field('key', pa.string()),
                pa.field('status', pa.string()),
                pa.field('error_message', pa.string())
            ]
            schema = pa.schema(fields)

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )

        successes = 0
        failed_to_subsample = 0

        decoder_kwargs = {"n_frames": self.n_frames+1, "fps": self.fps, "num_threads": 12, "return_bytes": True, "tmpdir": "/tmp/"}
        if self.batched:
            print("HI!")
            print(self.subsampler_batch_size, flush=True)
            print(self.n_frames, flush=True)
            data_loader_batch_size = self.subsampler_batch_size // self.n_frames
            decoder_kwargs["pad_frames"] = True
        else:
            data_loader_batch_size = 1
            decoder_kwargs["pad_frames"] = False

        if shard.startswith("s3://"):
            shard = f"pipe:aws s3 cp {shard} -"
        print(data_loader_batch_size, flush=True)
        dset = get_video_dataset(
            urls=shard,
            batch_size=data_loader_batch_size,
            decoder_kwargs=decoder_kwargs,
            resize_size=None,
            crop_size=(150,300),
            enforce_additional_keys=[],
            keys_to_remove=["m4a"]
        )

        if self.batched:
            for batch in dset:
                keys = batch["__key__"]
                captions = batch.get("txt", [b""] * len(keys))
                metas = batch.get("json", [{}] * len(keys))

                # Process the batch of frames
                frames_batch = [np.array(frames) for frames in batch.get("mp4")]
                optical_flow, metrics_list, error_message = self.optical_flow_subsampler(frames_batch, n_frames_per_video=self.n_frames)
                print("Im having an error: ", error_message, flush=True)
                if error_message is not None:
                    for key, caption, meta in zip(keys, captions, metas):
                        print(key)
                        meta["status"] = "failed_to_subsample"
                        meta["error_message"] = error_message
                        sample_writer.write(
                            {}, 
                            key, 
                            caption, 
                            meta
                        )
                    continue
                for idx, (key, caption, meta, flows, metrics) in enumerate(zip(keys, captions, metas, optical_flow, metrics_list)):
                    status = "success"
                    meta["status"] = status
                    meta["mean_optical_flow_magnitude"] = float(metrics[0])
                    meta["mean_optical_flow_magnitude_per_frame"] = [float(x) for x in metrics[1]]
                    meta["optical_flow_fps"] = self.fps
                    if self.downsample_size:
                        meta["optical_flow_downsample_size"] = self.downsample_size
                    meta["optical_flow_dtype"] = str(self.dtype)

                    streams = {}
                    streams["numpy_metadata"] = batch.get("npz", [{}]*len(keys))[idx]
                    if isinstance(streams["numpy_metadata"], bytes):
                        npz_bytes = io.BytesIO(streams["numpy_metadata"])
                        streams["numpy_metadata"] = dict(np.load(npz_bytes))
                    streams["numpy_metadata"]["optical_flow"] = flows
                    streams["numpy_metadata"] = numpy_npz_dumps(streams["numpy_metadata"])

                    streams["video"] = batch.get("video_bytes", [b""]*len(keys))[idx]

                    sample_writer.write(
                        streams,
                        key,
                        caption,
                        meta,
                    )
        else:
            for sample in dset:
                key = sample["__key__"][0]
                caption = sample.get("txt", b"")[0]
                meta = sample.get("json", {})[0]

                streams = {}
                frames = np.array(sample.get("mp4")[0])
                native_fps = sample.get("native_fps").item()

                for mod, fmt in self.encode_formats.items():
                    streams[mod] = sample.get(fmt, b"")

                optical_flow, metrics, error_message = self.optical_flow_subsampler(frames)

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
                if self.downsample_size:
                    meta["optical_flow_downsample_size"] = self.downsample_size
                meta["optical_flow_dtype"] = str(self.dtype)

                streams["numpy_metadata"] = sample.get("npz", {})
                if isinstance(streams["numpy_metadata"], bytes):
                    npz_bytes = io.BytesIO(streams["numpy_metadata"])
                    streams["numpy_metadata"] = dict(np.load(npz_bytes))
                streams["numpy_metadata"]["optical_flow"] = optical_flow
                streams["numpy_metadata"] = numpy_npz_dumps(streams["numpy_metadata"])

                streams["video"] = sample.get("video_bytes")[0]
                for modality in streams:
                    if isinstance(streams[modality], list):
                        streams[modality] = streams[modality][0]

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


class OldOpticalFlowWorker:
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
        optical_flow_params,
        is_slurm_task,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.thread_count = thread_count
        self.encode_formats = encode_formats
        self.save_caption = True
        self.detector = optical_flow_params.get("detector", "cv2")
        self.fps = optical_flow_params.get("fps", -1)
        self.downsample_size = optical_flow_params.get("downsample_size", None)
        self.dtype = optical_flow_params.get("dtype", "fp16")
        self.detector_args = optical_flow_params.get("detector_args", None)

        self.optical_flow_subsampler = OpticalFlowSubsampler(
            detector=self.detector,
            args=self.detector_args,
            fps=self.fps,
            downsample_size=self.downsample_size,
            dtype=self.dtype,
            is_slurm_task=is_slurm_task,
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
        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")

            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception as e: # pylint: disable=broad-except
            fields = [
                pa.field('key', pa.string()),
                pa.field('status', pa.string()),
                pa.field('error_message', pa.string())
            ]
            schema = pa.schema(fields)

        status_dict = CappedCounter()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id, self.output_folder, self.save_caption, self.oom_shard_count, schema, self.encode_formats
        )

        successes = 0
        failed_to_subsample = 0

        decoder_kwargs = {"n_frames": self.n_frames, "fps": self.fps, "num_threads": 4, "return_bytes": True, "tmpdir": "/tmp/"}

        if shard.startswith("s3://"):
            shard = f"pipe:aws s3 cp {shard} -"

        dset = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs=decoder_kwargs,
            resize_size=None,
            crop_size=None,
            enforce_additional_keys=[],
            pad_frames=pad_frames
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
            if self.downsample_size:
                meta["optical_flow_downsample_size"] = self.downsample_size
            meta["optical_flow_dtype"] = str(self.dtype)

            streams["numpy_metadata"] = sample.get("npz", {})
            if isinstance(streams["numpy_metadata"], bytes):
                npz_bytes = io.BytesIO(streams["numpy_metadata"])
                streams["numpy_metadata"] = dict(np.load(npz_bytes))
            streams["numpy_metadata"]["optical_flow"] = optical_flow
            streams["numpy_metadata"] = numpy_npz_dumps(streams["numpy_metadata"])

            streams["video"] = sample.get("video_bytes")[0]
            for modality in streams:
                if isinstance(streams[modality], list):
                    streams[modality] = streams[modality][0]

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
