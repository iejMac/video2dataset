"""classes and functions for downloading videos"""
import requests
import tempfile
import yt_dlp


# TODO make this better / audio support
def get_format_selector(format_args, retry):
    """
    Gets format selector based on retry number.
    """
    def format_selector(ctx):
        formats = ctx.get("formats")
        if retry == 0:
            for f in formats:
                if f.get("format_note", None) != QUALITY:
                    continue
                break
        else:
            for f in formats:  # take WORST video format available
                if f.get("vcodec", None) == "none":
                    continue
                break
        yield {
            "format_id": f["format_id"],
            "ext": f["ext"],
            "requested_formats": [f],
            "protocol": f["protocol"],
        }

    return format_selector


def handle_youtube(youtube_url, format_args, retry):
    """returns file and destination name from youtube url."""
    ydl_opts = {
        "quiet": True,
        "format": get_format_selector(format_args, retry),
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info = ydl.extract_info(youtube_url, download=False)
    formats = info.get("requested_formats", None)
    f = formats[0]

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


def handle_url(url):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - fname to save frames to.
    """
    if "youtube" in url:  # youtube link
        file, name = handle_youtube(url)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, name = handle_mp4_link(url)
    else:
        print("Warning: Incorrect URL type")
        return None, None, ""

    return file.name, file, name


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self) -> None:
        pass

    def __call__(self, row, timeout, retries):
        key, url = row
        file_name, file, _ = handle_url(url)
        with open(file_name, "rb") as vid_file:
            vid_bytes = vid_file.read()
        if file is not None:  # for python files that need to be closed
            file.close()
        return key, vid_bytes, None
