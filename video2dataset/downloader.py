"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp
from .audio_utils import get_info_and_resample


QUALITY = "360p"


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""
    ydl_opts = {}
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    f = None
    for f in formats:
        if f.get("format_note", None) != QUALITY:
            continue
        break

    cv2_vid = f.get("url", None)
    dst_name = info.get("id") + ".npy"
    return cv2_vid, dst_name


def handle_mp4_link(
    mp4_link: str,
    extract_audio: bool = False,
    sample_rate: int = None
):
    if extract_audio:
        audio_stream, audio_info = get_info_and_resample(
            mp4_link, sample_rate)  # returns bytes of resampled audio and audio info
        return None, audio_stream, audio_info, None

    else:
        resp = requests.get(mp4_link, stream=True)
        ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
        ntf.write(resp.content)
        ntf.seek(0)
        dst_name = mp4_link.split("/")[-1][:-4] + ".npy"
        return ntf.name, ntf, None, dst_name


def handle_url(url, sample_rate=None, get_audio=False):
    """
    Input:
        url: url of video
        get_audio: to extract audio from video
        sample_rate: desired sample rate of the output audio

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - numpy fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        load_file, name = handle_youtube(url)
        return load_file, None, name
        # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4") or get_audio:  # mp4 link
        f_name, file, info, name = handle_mp4_link(
            url, get_audio, sample_rate=sample_rate)
        return file, name, f_name, info
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""


class Downloader:
    def __init__(self):
        pass
