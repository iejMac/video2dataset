"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp


def handle_youtube(youtube_url, dl_timeout, video_height, video_width):
    """returns file and destination name from youtube url."""
    ydl_opts = {  # TODO: specify height width here and and just extract requested format
        "quiet": True,
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    filtered_formats = [
        f for f in formats if f["height"] is not None and f["height"] >= video_height and f["width"] >= video_width
    ]
    vid_url = filtered_formats[0].get("url")

    # For video2dataset we need the bytes:
    ntf, _ = handle_mp4_link(vid_url, dl_timeout)
    return ntf, None


def handle_mp4_link(mp4_link, dl_timeout):
    resp = requests.get(mp4_link, stream=True, timeout=dl_timeout)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    return ntf, None


def handle_url(url, dl_timeout, format_args):
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
            file, error_message = handle_youtube(url, dl_timeout, **format_args)
        except Exception as e:  # pylint: disable=(broad-except)
            file, error_message = None, str(e)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, error_message = handle_mp4_link(url, dl_timeout)
    else:
        file, error_message = None, "Warning: Incorrect URL type"
    return file, error_message


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_height, video_width, dl_timeout) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
        }
        self.dl_timeout = dl_timeout

    def __call__(self, row):
        key, url = row
        file, error_message = handle_url(url, self.dl_timeout, self.format_args)
        if error_message is None:
            with open(file.name, "rb") as vid_file:
                vid_bytes = vid_file.read()
            file.close()
        else:
            vid_bytes = None
        return key, vid_bytes, error_message
