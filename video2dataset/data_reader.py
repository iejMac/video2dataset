"""classes and functions for downloading videos"""
import os

import uuid
import requests
import tempfile
import yt_dlp


def handle_youtube(youtube_url, tmp_dir, video_height, video_width):
    """returns file and destination name from youtube url."""
    path = f"{tmp_dir}/{str(uuid.uuid4())}.mp4"
    format_string = (
        f"bv*[height<={video_height}][width<={video_width}][ext=mp4]"
        + f"+ba[ext=m4a]/b[height<={video_height}][width<={video_width}]"
    )
    ydl_opts = {
        "outtmpl": path,
        "format": format_string,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_url)
    return path, None


def handle_mp4_link(mp4_link, tmp_dir, dl_timeout):
    resp = requests.get(mp4_link, stream=True, timeout=dl_timeout)
    path = f"{tmp_dir}/{str(uuid.uuid4())}.mp4"
    with open(path, "wb") as f:
        f.write(resp.content)
    return path, None


def handle_url(url, dl_timeout, format_args, tmp_dir):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        try:
            file, error_message = handle_youtube(url, tmp_dir, **format_args)
        except Exception as e:  # pylint: disable=(broad-except)
            file, error_message = None, str(e)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, error_message = handle_mp4_link(url, tmp_dir, dl_timeout)
    else:
        file, error_message = None, "Warning: Incorrect URL type"
    return file, error_message


class VideoDataReader:
    """Video data reader provide data for a video"""
    def __init__(self, video_height, video_width, dl_timeout, tmp_dir) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
        }
        self.dl_timeout = dl_timeout
        self.tmp_dir = tmp_dir

    def __call__(self, row):
        key, url = row
        file_path, error_message = handle_url(url, self.dl_timeout, self.format_args, self.tmp_dir)
        if error_message is None:
            with open(file_path, "rb") as vid_file:
                vid_bytes = vid_file.read()
        else:
            vid_bytes = None

        if file_path is not None: # manually remove tempfile
            os.remove(file_path)
        return key, vid_bytes, error_message
