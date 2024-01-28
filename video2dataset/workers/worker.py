"""Standard worker for video2dataset."""
from dataclasses import dataclass, field
import ffmpeg
import numpy as np
import os
import tempfile
from typing import Any, List, Tuple, Optional, Literal, cast
import uuid

from video2dataset.logger import CappedCounter
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
    Subsampler,
)
from video2dataset.types import EncodeFormats, Streams, Metadata, TempFilepaths


@dataclass
class Subsamplers:
    """Subsamplers used in processing"""

    ffprobe_subsampler: Optional[FFProbeSubsampler] = None
    modal_subsamplers: dict = field(default_factory=dict)
    cut_detection_subsampler: Optional[CutDetectionSubsampler] = None
    cuts_are_clips: bool = False
    broadcast_subsampler: Subsampler = field(default_factory=NoOpSubsampler)


def get_subsamplers(
    config: dict,
    input_encode_formats: EncodeFormats,
    do_clipping: bool = False,
) -> Tuple[Subsamplers, EncodeFormats]:
    """Initialize all subsamplers using config"""

    clipping_subsampler = ClippingSubsampler(
        oom_clip_count=5,
        encode_formats=input_encode_formats,
        **config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
    )
    need_keyframes = clipping_subsampler.precision == "keyframe_adjusted"

    cut_detection_subsampler = None
    cuts_are_clips = False
    if "CutDetectionSubsampler" in config["subsampling"]:
        if "args" in config["subsampling"]["CutDetectionSubsampler"]:
            cut_detection_subsampler = CutDetectionSubsampler(**config["subsampling"]["CutDetectionSubsampler"]["args"])
        cuts_are_clips = config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

    broadcast_subsampler = (
        clipping_subsampler
        if (do_clipping or config["storage"]["captions_are_subtitles"] or cuts_are_clips)
        else NoOpSubsampler()
    )

    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(**config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"])
        ffprobe_subsampler.extract_keyframes |= need_keyframes

    video_subsamplers: List[Any] = []
    if "ResolutionSubsampler" in config["subsampling"]:
        video_subsamplers.append(ResolutionSubsampler(**config["subsampling"]["ResolutionSubsampler"]["args"]))
    if "FrameSubsampler" in config["subsampling"]:
        video_subsamplers.append(FrameSubsampler(**config["subsampling"]["FrameSubsampler"]["args"]))

    audio_subsamplers: List[Any] = []
    if "AudioRateSubsampler" in config["subsampling"]:
        audio_subsamplers.append(AudioRateSubsampler(**config["subsampling"]["AudioRateSubsampler"]["args"]))

    modal_subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    # output encoding formats
    output_encode_formats = input_encode_formats.copy()
    if modal_subsamplers["audio"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["audio"]}) == 1
        )  # assert that all audio subsamplers have the same output format
        output_encode_formats["audio"] = modal_subsamplers["audio"][0].encode_format
    if modal_subsamplers["video"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["video"]}) == 1
        )  # assert that all video subsamplers have the same output format
        output_encode_formats["video"] = modal_subsamplers["video"][0].encode_format

    return (
        Subsamplers(
            ffprobe_subsampler=ffprobe_subsampler,
            modal_subsamplers=modal_subsamplers,
            cut_detection_subsampler=cut_detection_subsampler,
            cuts_are_clips=cuts_are_clips,
            broadcast_subsampler=broadcast_subsampler,
        ),
        output_encode_formats,
    )


@dataclass
class ShardStatus:
    """Shard processing status"""

    successes: int = 0
    failed: dict = field(
        default_factory=lambda: {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
    )
    status_dict: CappedCounter = field(default_factory=CappedCounter)
    error_message: Optional[str] = None
    count: int = 0
    bytes_downloaded: int = 0


def extract_video_metadata(
    subsamplers: Subsamplers,
    shard_status: ShardStatus,
    metadata: Metadata,
    video_filepath: str,
    captions_are_subtitles: bool,
):
    """Add additional metadata keys for video file"""

    if subsamplers.ffprobe_subsampler is not None:
        metadata, shard_status.error_message = subsamplers.ffprobe_subsampler(video_filepath, metadata)
        assert shard_status.error_message is None

    if captions_are_subtitles:  # create clips
        subtitles = metadata["yt_meta_dict"]["subtitles"]
        metadata["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
    elif subsamplers.cut_detection_subsampler is not None:  # apply cut detection to get clips
        metadata, shard_status.error_message = subsamplers.cut_detection_subsampler(video_filepath, metadata)
        assert shard_status.error_message is None
        cuts = metadata["cuts"]
        assert cuts is not None
        if subsamplers.cuts_are_clips:
            metadata["clips"] = (np.array(cuts["cuts_original_fps"]) / cuts["original_fps"]).tolist()

    return metadata


def process_sample(
    subsamplers: Subsamplers,
    shard_status: ShardStatus,
    streams: Streams,
    key: str,
    caption: str,
    metadata: Metadata,
    captions_are_subtitles: bool,
    shard_sample_writer: Any,  # TODO: type correctly
):
    """Process a single video"""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # save temp stream dumps
            temp_filepaths: TempFilepaths = {}
            for modality in streams:
                modality = cast(Literal["video", "audio"], modality)
                temp_filepaths[modality] = []
                for stream in streams[modality]:
                    stream_uuid = str(uuid.uuid4())
                    temp_filepath = os.path.join(tmpdir, stream_uuid)
                    with open(temp_filepath, "wb") as f:
                        f.write(stream)
                    temp_filepaths[modality].append(temp_filepath)

            # this is pre-broadcast, so there should only be one video
            assert "video" in temp_filepaths
            assert len(temp_filepaths["video"]) == 1
            video_filepath = temp_filepaths["video"][0]

            # add info about keyframes and cuts
            metadata = extract_video_metadata(
                subsamplers=subsamplers,
                shard_status=shard_status,
                metadata=metadata,
                video_filepath=video_filepath,
                captions_are_subtitles=captions_are_subtitles,
            )

            # process video modality
            video_filepaths = temp_filepaths["video"]
            for video_filepath in video_filepaths:
                # prep first node
                ffmpeg_node = ffmpeg.input(video_filepath)

                # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                (
                    ffmpeg_node,
                    metadatas,
                    shard_status.error_message
                ) = subsamplers.broadcast_subsampler(
                    ffmpeg_node=ffmpeg_node,
                    tmpdir=tmpdir,
                    metadatas=[metadata],
                )
                if shard_status.error_message is not None:
                    metadata["clips"] = []
                    assert False

        for modality in list(subsampled_streams.keys()):
            for modality_subsampler in subsamplers.modal_subsamplers[modality]:
                subsampled_streams, metadatas, shard_status.error_message = modality_subsampler(
                    subsampled_streams, metadatas
                )
                assert shard_status.error_message is None

        shard_status.successes += 1
        status = "success"
        shard_status.status_dict.increment(status)

        subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
        if len(subsampled_streams_list) == 0:  # no audio or video, just write metadata
            metadata["status"] = status
            shard_sample_writer.write(
                {},
                key,
                caption,
                metadata,
            )
            return
        for subsampled_streams, subsampled_metadata in zip(subsampled_streams_list, metadatas):
            subsampled_metadata["status"] = status
            text_caption = caption
            if captions_are_subtitles:
                clip_subtitles = subsampled_metadata.get("clip_subtitles")
                first_clip_subtitles = clip_subtitles[0] if clip_subtitles else None
                subtitle_lines = first_clip_subtitles["lines"] if first_clip_subtitles else None
                text_caption = subtitle_lines[0] if subtitle_lines else text_caption
            shard_sample_writer.write(
                subsampled_streams,
                subsampled_metadata["key"],
                text_caption,
                subsampled_metadata,
            )
    except Exception as err:  # pylint: disable=broad-except
        print(err)
        shard_status.failed["failed_to_subsample"] += 1
        shard_status.status_dict.increment(shard_status.error_message)
        metadata["status"] = "failed_to_subsample"
        metadata["error_message"] = shard_status.error_message
        shard_sample_writer.write(
            {},
            key,
            caption,
            metadata,
        )
