"""classes and functions for downloading videos"""
import os
import uuid
import requests
import yt_dlp


def get_yt_meta(url, yt_metadata_args: dict) -> dict:
    """Return info dict and/or downloads subtitles
        yt_metadata_args is a dict of follwing format:
        yt_metadata_args = {
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'subtitles_dir': 'subtitles',
            'writeautomaticsub': True,
            'get_info': True
        }

        writesubtitles:    Write the video subtitles to a file
        writeautomaticsub: Write the automatically generated subtitles to a file
        subtitleslangs:    List of languages of the subtitles to download (can be regex). The list may contain "all" to refer to all the available
                            subtitles. 
        subtitles_dir: a folder to save subtitles to (will be created if doesn't exist)
        get_info: whether to add info (title, description, tags etc) to the output.

    """

    subtitles_dir = yt_metadata_args.get('subtitles_dir', None)

    if subtitles_dir:
        if not os.path.exists(subtitles_dir):
            os.makedirs(subtitles_dir)
        yt_metadata_args['outtmpl'] = subtitles_dir + '/'+'%(id)s.%(ext)s'

    yt_metadata_args['skip_download'] = True
    yt_metadata_args['ignoreerrors'] = True
    yt_metadata_args['quiet'] = True

    info_dict = None

    with yt_dlp.YoutubeDL(yt_metadata_args) as yt:
        yt.download([url])
        if yt_metadata_args['get_info']:
            info_dict = yt.extract_info(url, download=False)
            info_dict.pop('subtitles')
            info_dict.pop('requested_formats')
            info_dict.pop('formats')
            info_dict.pop('thumbnails')
            info_dict.pop('automatic_captions')
    return info_dict


def handle_youtube(youtube_url, tmp_dir, yt_metadata_args, video_height, video_width):
    """returns file and destination name from youtube url."""
    path = f"{tmp_dir}/{str(uuid.uuid4())}.mp4"
    format_string = (
        f"bv*[height<={video_height}][width<={video_width}][ext=mp4]"
        + f"+ba[ext=m4a]/b[height<={video_height}][width<={video_width}]"
    )
    ydl_opts = {
        "outtmpl": path,
        "format": format_string,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_url)
    if yt_metadata_args:
        info_dict = get_yt_meta(youtube_url, yt_metadata_args)
    else:
        info_dict = None
    return path, info_dict, None


def handle_mp4_link(mp4_link, tmp_dir, dl_timeout):
    resp = requests.get(mp4_link, stream=True, timeout=dl_timeout)
    path = f"{tmp_dir}/{str(uuid.uuid4())}.mp4"
    with open(path, "wb") as f:
        f.write(resp.content)
    return path, None


def handle_url(url, dl_timeout, format_args, tmp_dir, yt_metadata_args=None):
    """
    Input:
        url: url of video

    Output:
        load_file - variable used to load video.
        file - the file itself (in cases where it needs to be closed after usage).
        name - fname to save frames to.
    """

    info_dict = None
    if "youtube" in url:  # youtube link
        try:
            file, info_dict, error_message = handle_youtube(
                url, tmp_dir, yt_metadata_args, **format_args)
        except Exception as e:  # pylint: disable=(broad-except)
            file, info_dict, error_message = None,  None, str(e)
    # TODO: add .avi, .webm, should also work
    elif url.endswith(".mp4"):  # mp4 link
        file, error_message = handle_mp4_link(url, tmp_dir, dl_timeout)
    else:
        file, error_message = None, "Warning: Incorrect URL type"
    return file, error_message, info_dict


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_height, video_width, dl_timeout, tmp_dir, yt_meta_args) -> None:
        self.format_args = {
            "video_height": video_height,
            "video_width": video_width,
        }
        self.dl_timeout = dl_timeout
        self.tmp_dir = tmp_dir
        self.yt_meta_args = yt_meta_args

    def __call__(self, row):
        key, url = row
        file_path, error_message, info_dict = handle_url(
            url, self.dl_timeout, self.format_args, self.tmp_dir, self.yt_meta_args)
        if error_message is None:
            with open(file_path, "rb") as vid_file:
                vid_bytes = vid_file.read()
        else:
            vid_bytes = None

        if file_path is not None:  # manually remove tempfile
            os.remove(file_path)
        return key, vid_bytes, info_dict, error_message
