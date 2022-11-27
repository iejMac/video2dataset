"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp

from timeout_decorator import timeout, TimeoutError  # pylint: disable=redefined-builtin
from yt_dlp.utils import DownloadError


def get_fast_format(formats, find_format_timeout):
    """returns the closest format that downloads quickly"""

    @timeout(find_format_timeout)
    def check_speed(f):
        url = f.get("url")
        ntf, _ = handle_mp4_link(url, 10)
        with open(ntf.name, "rb") as vid_file:
            _ = vid_file.read()
        ntf.close()

    format_id = None
    for fmt in formats:
        try:
            check_speed(fmt)
            format_id = fmt.get("format_id")
            break
        except TimeoutError as _:
            pass

    return format_id


def handle_youtube(youtube_url, max_format_tries, dl_timeout, find_format_timeout, video_height, video_width):
    """returns file and destination name from youtube url."""
    # Probe download speed:
    ydl_opts = {
        "quiet": True,
        "external-download": "ffmpeg",
        "external-downloader-args": "ffmpeg_i:-ss 0 -t 2",  # download 2 seconds
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    filtered_formats = [
        f for f in formats if f["height"] is not None and f["height"] >= video_height and f["width"] >= video_width
    ]

    # TODO: how do we drop the video when format_id is None (all retires timed out)
    format_id = get_fast_format(filtered_formats[:max_format_tries], find_format_timeout)
    if format_id is None:
        return None, "No format available given input constraints"

    # Get actual video:
    # TODO: figure out a way of just requesting the format by format_id
    ydl_opts = {"quiet": True}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    f = [f for f in formats if f["format_id"] == format_id][0]

    vid_url = f.get("url", None)

    # For video2dataset we need the bytes:
    ntf, _ = handle_mp4_link(vid_url, dl_timeout)
    return ntf, None


def handle_mp4_link(mp4_link, dl_timeout):
    resp = requests.get(mp4_link, stream=True, timeout=dl_timeout)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    return ntf, None


def handle_url(url, max_format_tries, dl_timeout, find_format_timeout, format_args):
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
            file, error_message = handle_youtube(url, max_format_tries, dl_timeout, find_format_timeout, **format_args)
        except DownloadError as e:
            file, error_message = None, str(e)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, error_message = handle_mp4_link(url, dl_timeout)
    else:
        file, error_message = None, "Warning: Incorrect URL type"
    return file, error_message


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_height, video_width, dl_timeout, find_format_timeout, max_format_tries) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
        }
        self.dl_timeout = dl_timeout
        self.find_format_timeout = find_format_timeout
        self.max_format_tries = max_format_tries

    def __call__(self, row):
        key, url = row
        file, error_message = handle_url(
            url, self.max_format_tries, self.dl_timeout, self.find_format_timeout, self.format_args
        )
        if error_message is None:
            with open(file.name, "rb") as vid_file:
                vid_bytes = vid_file.read()
            file.close()
        else:
            vid_bytes = None
        return key, vid_bytes, error_message
