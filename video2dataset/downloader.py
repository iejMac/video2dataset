"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp
from .audio_utils import get_info_and_resample
import requests


QUALITY = "360p"


def handle_youtube(youtube_url: str, get_audio: bool, sample_rate: int):
    """returns file and destination name from youtube url."""
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    }

    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio/best',
    }
    try:
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info = ydl.extract_info(youtube_url, download=False)
        formats = info.get("formats", None)

        if get_audio:

            for f in formats:
                if f.get("format_note", None) != QUALITY and not f.get("asr", None):
                    continue
                break

        else:
            for f in formats:
                if f.get("format_note", None) != QUALITY:
                    continue
                break

        video_info = {
            'id': info['id'],
            'title': info['title'],
            'description': info.get('description', None),
            'tags': info.get('tags', None)
        }
        audio_info = {
            'format': 'mp3',
            'duration': info.get('duration', None),
            'channels': info.get('audio_channels', None),
            'sample_rate': info.get('asr', None)
        }
        video_info['audio_info'] = audio_info
        url = f.get('url', None)
        if get_audio and url:
            stream, audio_info = get_info_and_resample(
                url, sample_rate)
        else:
            res = requests.get(url, headers=headers, stream=True)
            stream = res.content
        dst_name = info.get("id") + ".npy"

        return stream, None, dst_name, video_info, None
    except Exception as err:
        return None, None, None, None, str(err)


def handle_mp4_link(
    mp4_link: str,
    extract_audio: bool = False,
    sample_rate: int = None
):

    try:
        if extract_audio:
            audio_stream, audio_info = get_info_and_resample(
                mp4_link, sample_rate)  # returns bytes of resampled audio and audio info
            return None, audio_stream, audio_info, None, None

        else:
            resp = requests.get(mp4_link, stream=True)
            ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
            ntf.write(resp.content)
            ntf.seek(0)
            dst_name = mp4_link.split("/")[-1][:-4] + ".npy"
            return ntf.name, ntf, None, dst_name, None

    except Exception as err:
        return None, None, None, None, str(err)


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

    if "youtube" in url or "youtu.be" in url:  # youtube link
        load_file, f_name, dst_name,  info, err = handle_youtube(
            url, get_audio, sample_rate)
        return load_file, f_name, dst_name, info

        # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4") or get_audio:  # mp4 link
        f_name, file, info, dst_name, err = handle_mp4_link(
            url, get_audio, sample_rate=sample_rate)
        return file, f_name, dst_name, info

    else:
        print("Warning: Incorrect URL type")
        return None, None, ""


class Downloader:
    def __init__(self):
        pass
