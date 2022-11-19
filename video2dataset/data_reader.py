"""classes and functions for downloading videos"""
import requests
import tempfile
import time
import yt_dlp



def handle_youtube(youtube_url, retries, timeout, video_height, video_width):
    """returns file and destination name from youtube url."""
    # Probe download speed:
    ydl_opts = {
        "quiet": True,
        "external-download": "ffmpeg",
        "external-downloader-args": "ffmpeg_i:-ss 0 -t 2", # download 2 seconds
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    filtered_formats = [f for f in formats if f['height'] is not None and f['height'] >= video_height and f['width'] >= video_width]

    format_id = None
    # TODO: stop downloading after timeout (had a case where it took 46s)
    for f in filtered_formats[:retries + 1]:
        t0 = time.time()
        ntf, _ = handle_mp4_link(f["url"])
        with open(ntf.name, "rb") as vid_file:
            vid_bytes = vid_file.read()
        ntf.close()
        tf = time.time()
        print(f"{tf - t0} | {timeout}")
        if tf - t0 < timeout:
            format_id = f["format_id"]
            break

    # Get actual video:
    # TODO: figure out a way of just requesting the format by format_id
    ydl_opts = {
        "quiet": True,
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    f = [f for f in formats if f['format_id'] == format_id][0]

    vid_url = f.get("url", None)
    dst_name = info.get("id")

    # For video2dataset we need the bytes:
    ntf, _ = handle_mp4_link(vid_url)
    return ntf, dst_name


def handle_mp4_link(mp4_link):
    resp = requests.get(mp4_link, stream=True)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    dst_name = mp4_link.split("/")[-1][:-4]
    return ntf, dst_name


def handle_url(url, retries, timeout, format_args):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        file, name = handle_youtube(url, retries, timeout, **format_args)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""

    return file.name, file, name


class VideoDataReader:
    """Video data reader provide data for a video"""
    def __init__(self, video_height, video_width, timeout, retries) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
        }
        self.timeout = timeout
        self.retries = retries

    def __call__(self, row):
        key, url = row
        file_name, file, _ = handle_url(url, self.retries, self.timeout, self.format_args)
        with open(file_name, "rb") as vid_file:
            vid_bytes = vid_file.read()
        file.close()
        return key, vid_bytes, None
