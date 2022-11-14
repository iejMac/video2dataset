"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp
import requests
import json
from subprocess import check_output
import os
import subprocess
import shlex


def get_media_info(filename: str) -> dict:
    """Extracts audio format, number of channels, duration and sample rate

    Keyword arguments:
    filename - video filename or URL

    Returns:
    dict of info
    """

    result = check_output(['ffprobe',
                           '-hide_banner', '-loglevel', 'panic',
                           '-select_streams',
                           'a:0',
                           '-show_streams',
                           '-of',
                           'json', filename])

    result = json.loads(result)['streams']
    result = result[0]
    codec_name = result.get('codec_name', None)
    channels = result.get('channels', None)
    duration = result.get('duration', None)
    sample_rate = result.get('sample_rate', None)

    return {
        'format': codec_name,
        'channels': channels,
        'duration': duration,
        'orig_sample_rate': sample_rate
    }


def get_info_and_resample(url: str, sample_rate: int) -> tuple:
    """Changes sample rate of an audio if sample rate is provided and extracts audio info

    Keyword arguments:
    url - video filename or url
    filename - desired filename of audio
    sample_rate - desired sample rate
    """

    headers = '"User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36"'
    media_info = get_media_info(url)
    media_info['sample_rate'] = sample_rate
    video_info = {
        'audio_info': media_info
    }

    sample_rate = sample_rate if sample_rate else media_info['sample_rate']

    command = f'ffmpeg -headers {headers}  -i {url} -vn -ac 2 -f wav -acodec pcm_s16le -ar {sample_rate} - -hide_banner -loglevel panic'

    ffmpeg_cmd = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        shell=False
    )
    b = b''
    nsamples = 1024
    itemsize = 2
    while True:
        output = ffmpeg_cmd.stdout.read(nsamples*itemsize)
        if len(output) > 0:
            b += output
        else:
            error_msg = ffmpeg_cmd.poll()
            if error_msg is not None:
                break

    return b, video_info


QUALITY = "360p"


def handle_youtube(youtube_url: str, video_format: str, sample_rate: int):
    """returns stream and video/audio info from youtube url."""
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
    }

    ydl_opts = {
        'quiet': True,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=mp3]/mp4',
    }

    streams = dict()
    try:
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info = ydl.extract_info(youtube_url, download=False)

        formats = info.get("formats", None)
        video_info = {
            'id': info['id'],
            'title': info['title'],
            'description': info.get('description', None),
            'tags': info.get('tags', None)
        }
        for vf in video_format.split(','):
            if vf == 'mp3':

                for f in formats:
                    if f.get("audio_ext", None) != 'm4a' and not f.get("asr", None):
                        continue
                    break

                url = f.get('url', None)

                stream, audio_info = get_info_and_resample(
                    url, sample_rate)
                streams[vf] = dict()
                streams[vf]['file'] = stream
                video_info['audio_info'] = audio_info

            else:
                for f in formats:
                    if f.get("format_note", None) != QUALITY:
                        continue
                    break
                url = f.get('url', None)
                res = requests.get(url, headers=headers, stream=True)
                stream = res.content
                streams[vf] = dict()
                streams[vf]['file'] = stream

        streams['info'] = video_info
        streams['error'] = ''
    except Exception as err:
        streams['info'] = {}
        streams['error'] = err
    return streams


def handle_mp4_link(
    mp4_link: str,
    video_format: str,
    sample_rate: int = None
):
    streams = dict()
    try:
        for vf in video_format.split(','):
            if vf == 'mp3':
                audio_stream, audio_info = get_info_and_resample(
                    mp4_link, sample_rate)
                streams[vf] = dict()
                streams[vf]['file'] = audio_stream

            else:
                resp = requests.get(mp4_link, stream=True)
                streams[vf] = dict()
                streams[vf]['file'] = resp.content
        streams['info']['audio_info'] = audio_info
        streams['error'] = ''
        return streams

    except Exception as err:
        streams['error'] = str(err)
        return streams


def handle_url(url, video_format, sample_rate=None):
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
        streams = handle_youtube(
            url, video_format, sample_rate)
        return streams

        # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4") or url.endswith(".mp3") or url.endswith(".m4a"):  # mp4 link
        streams = handle_mp4_link(
            url, video_format, sample_rate=sample_rate)
        return streams

    else:
        print("Warning: Incorrect URL type")
        return None, None, ""


class Downloader:
    def __init__(self):
        pass
