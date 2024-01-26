"""extracts basic video compression metadata."""
import json
import subprocess
from typing import Tuple, Optional

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error


# TODO: figuer out why this is so slow (12 samples/s)
class FFProbeSubsampler(Subsampler):
    """
    Extracts metadata from bytes.
    Args:
        extract_keyframes (bool): Whether to extract keyframe timestamps.
    """

    def __init__(self, extract_keyframes=False):
        self.extract_keyframes = extract_keyframes

    def __call__(self, video_filepath: str, metadata: Optional[Metadata] = None) -> Tuple[Metadata, Error]:
        metadata = metadata if metadata is not None else {}
        try:
            # extract video metadata
            command = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                f"{video_filepath}",
            ]
            if self.extract_keyframes:
                command.extend(["-select_streams", "v:0", "-show_entries", "packet=pts_time,flags"])
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            video_metadata = json.loads(process.stdout)

            # extract keyframe timestamps if requested
            if self.extract_keyframes:
                keyframe_timestamps = [
                    float(packet["pts_time"])
                    for packet in video_metadata["packets"]
                    if "K" in packet.get("flags", "")
                ]
                if "duration" in video_metadata["format"]:
                    duration = float(video_metadata["format"]["duration"])
                    keyframe_timestamps.append(duration)
                video_metadata["keyframe_timestamps"] = keyframe_timestamps

            # save and return metadata
            video_metadata.pop("packets")  # Don't need it anymore
            metadata["video_metadata"] = video_metadata
        except Exception as err:  # pylint: disable=broad-except
            return metadata, str(err)
        return metadata, None
