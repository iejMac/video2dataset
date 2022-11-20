"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp


QUALITY = "360p"


def handle_youtube(youtube_url):
    """returns file and destination name from youtube url."""

    # it selects best video in mp4 format but no better than 480p
    # or the worst video if there is no video under 360p with the best audio in m4a format
    ydl_opts = {
        'quiet': True,
        'format': 'bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]'
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    cv2_vid = info['url']
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
