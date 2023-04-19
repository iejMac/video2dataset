"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp


QUALITY = "360p"


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""
    yt_dlp.utils.std_headers['Referer'] = "https://www.youtube.com/"
    yt_dlp.utils.std_headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'

    ydl_opts = {
        'quiet': True,
        'format': 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]'
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("formats", None)
    cv2_vid = [f for f in formats if f['format_note']
               == QUALITY and f['ext'] == 'mp4'][0]['url']
    dst_name = info.get("id") + ".npy"
    return cv2_vid, dst_name


def handle_mp4_link(mp4_link):
    resp = requests.get(mp4_link, stream=True)
    ntf = tempfile.NamedTemporaryFile()  # pylint: disable=consider-using-with
    ntf.write(resp.content)
    ntf.seek(0)
    dst_name = mp4_link.split("/")[-1][:-4] + ".npy"
    return ntf, dst_name


def handle_url(url):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - numpy fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        load_file, name = handle_youtube(url)
        return load_file, None, name
        # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
        return file.name, file, name
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""


class Downloader:
    def __init__(self):
        pass
