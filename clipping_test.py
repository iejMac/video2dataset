import subprocess
import numpy as np
from video2dataset.subsamplers import ClippingSubsampler, CutDetectionSubsampler


def adjust_ranges_to_keyframes(ranges, keyframes):
    adjusted_ranges = []

    for start, end in ranges:
        # Find keyframes within this range
        keyframes_in_range = [k for k in keyframes if start <= k <= end]

        if keyframes_in_range:
            # If there are keyframes in the range, replace the range with the smallest
            # and largest keyframes
            adjusted_start = min(keyframes_in_range)
            adjusted_end = max(keyframes_in_range)
            if adjusted_start != adjusted_end:
                # Only add the range if the start and end times are different
                adjusted_ranges.append((adjusted_start, adjusted_end))
        else:
            # If no keyframes in the range, it's impossible to adjust it
            print(f"No keyframes in range ({start}, {end}), skipping")

    return adjusted_ranges


def get_keyframe_timestamps(video_path):
    command = [
        'ffprobe',
        '-loglevel', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'packet=pts_time,flags',
        '-of', 'csv=print_section=0',
        video_path      
    ]

    process1 = subprocess.Popen(command, stdout=subprocess.PIPE)
    process2 = subprocess.Popen(['awk', '-F,', '/K/ {print $1}'], stdin=process1.stdout, stdout=subprocess.PIPE)

    output, _ = process2.communicate()
    output = output.decode().strip()
    keyframes = [float(val) for val in output.split('\n')]

    # Get the video duration and append to the list
    command_duration = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    process_duration = subprocess.Popen(command_duration, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration, _ = process_duration.communicate()
    duration = duration.decode().strip()

    # Append the duration to the end of the list
    keyframes.append(float(duration))

    return keyframes

# Use function
video_path = "input.mp4"
cd = CutDetectionSubsampler("all", threshold=35)
clip = ClippingSubsampler(5, {"video":"mp4"}, 0.0, 9999, "all", False)

keyframe_timestamps = get_keyframe_timestamps(video_path)
print(keyframe_timestamps)
print(len(keyframe_timestamps))

meta = {"key": 0}
with open(video_path, "rb") as f:
    streams = {"video": [f.read()]}


print("Detecting cuts...")
_, cuts, _ = cd(streams, meta)
native_fps = cuts["original_fps"]
clips = (np.array(cuts["cuts_original_fps"]) / native_fps).tolist()
clips = [(round(x, 3), round(y, 3)) for (x, y) in clips if y - x >= 4.0]
print("CUTS:")
print(clips)
print(len(clips))

print("AADJUSTED CUTS:")
adjusted_clips = adjust_ranges_to_keyframes(clips, keyframe_timestamps)
print(adjusted_clips)
print(len(adjusted_clips))
clips = adjusted_clips

print("Clipping...")
meta["clips"] = clips
streams, metas, _ = clip(streams, meta)

print(metas)

print(len(streams["video"]))
for i, cl_bytes in enumerate(streams["video"]):
    with open("asdf/" + metas[i]["key"] + ".mp4", "wb") as f:
        f.write(cl_bytes)
