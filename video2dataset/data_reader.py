"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp
import subprocess
import shlex

from timeout_decorator import timeout, TimeoutError  # pylint: disable=redefined-builtin


def resample(url: str, sample_rate: int) -> tuple:
    """Changes sample rate of an audio if sample rate is provided and extracts audio info

    Keyword arguments:
    url - video filename or url
    filename - desired filename of audio
    sample_rate - desired sample rate
    """

    headers = '"User-Agent: Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36"'

    sample_rate_str = f'-ar {sample_rate}' if sample_rate else ''

    headers_str = f'-headers {headers}' if 'http' in url else ''

    command = f'ffmpeg {headers_str} -i {url} -vn -ac 2 -f wav -acodec pcm_s16le {sample_rate_str} - -hide_banner -loglevel panic'

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

    return b


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


def handle_youtube(youtube_url, max_format_tries, dl_timeout, find_format_timeout, video_height, video_width, video_format, sample_rate):
    """returns file and destination name from youtube url."""

    yt_metadata_args = {
        'writesubtitles': True,
        'allsubtitles': False,
        'subtitleslangs': ['en'],
        'subtitles_dir': 'subtitles',
        'dump_single_json': True

    }

    streams = dict()
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
    for vf in video_format:
        print(vf)
        if vf == "mp3":
            try:
                audio_url = [f for f in formats if f['format']
                             == '140 - audio only (medium)'][0]['url']
                streams['mp3'] = handle_mp3_link(audio_url, sample_rate)
                print(audio_url)
                err = None
            except Exception as err:
                print(err)
                streams["mp3"] = None
                err = str(err)
        else:

            # TODO: how do we drop the video when format_id is None (all retires timed out)
            format_id = get_fast_format(
                filtered_formats[:max_format_tries], find_format_timeout)
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
            streams["mp4"] = ntf
    print('-------')
    print(streams.keys())
    return streams, err


def handle_mp4_link(mp4_link, dl_timeout):
    resp = requests.get(mp4_link, stream=True, timeout=dl_timeout)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    return ntf, None


def handle_mp3_link(mp3_link, sample_rate):
    stream = resample(mp3_link, sample_rate)
    return stream, None


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
        file, error_message = handle_youtube(
            url, max_format_tries, dl_timeout, find_format_timeout, **format_args)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, error_message = handle_mp4_link(url, dl_timeout)
    elif url.endswith(".mp3"):
        sample_rate = format_args['sample_rate']
        file, error_message = handle_mp3_link(url, sample_rate)
    else:
        file, error_message = None, "Warning: Incorrect URL type"
    return file, error_message


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_height, video_width, dl_timeout, find_format_timeout, max_format_tries, video_format, sample_rate) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
            "video_format": video_format,
            "sample_rate": sample_rate
        }
        self.dl_timeout = dl_timeout
        self.find_format_timeout = find_format_timeout
        self.max_format_tries = max_format_tries

    def __call__(self, row):
        key, url = row
        streams, error_message = handle_url(
            url, self.max_format_tries, self.dl_timeout, self.find_format_timeout, self.format_args
        )
        print(error_message)
        vid_file = streams["mp4"]
        if error_message is None:
            with open(vid_file.name, "rb") as vid_file:
                vid_bytes = vid_file.read()
            vid_file.close()
        else:
            vid_bytes = None
        streams["mp4"] = vid_bytes

        return key, streams, error_message
