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

    result = json.loads(result)['streams'][0]

    codec_name = result['codec_name']
    channels = result['channels']
    duration = result['duration']
    sample_rate = result['sample_rate']

    return {
        'format': codec_name,
        'channels': channels,
        'duration': duration,
        'sample_rate': sample_rate
    }


def get_info_and_resample(url: str, sample_rate: int, get_info: bool = True) -> tuple:
    """Changes sample rate of an audio if sample rate is provided and extracts audio info

    Keyword arguments:
    url - video filename or url
    filename - desired filename of audio
    sample_rate - desired sample rate
    get_info - to return audio info
    """

    headers = '"User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36"'
    media_info = get_media_info(url)
    video_info = {
        'audio_info': media_info
    }
    sample_rate = sample_rate if sample_rate else media_info['sample_rate']

    command = f'ffmpeg -headers {headers}  -i {url} -vn -ac 2 -f wav -acodec pcm_s16le -ar {sample_rate} - -hide_banner -loglevel error'
    ffmpeg_cmd = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        shell=False
    )
    b = b''
    while True:
        output = ffmpeg_cmd.stdout.read()
        if len(output) > 0:
            b += output
        else:
            error_msg = ffmpeg_cmd.poll()
            if error_msg is not None:
                break
    if str(error_msg) == '0':
        error_msg = None
    else:
        error_msg = str(error_msg)
    if get_info:
        return b, video_info
    return b, None
