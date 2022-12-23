"""classes and functions for downloading videos"""
import os
import uuid
import requests
import yt_dlp
import io

try:
    import webvtt  # pip install webvtt-py
except ImportError as err:
    print(err)


def sub_to_dict(sub, dedupe=True, single=False) -> list:
    """Convert WebVTT to JSON, optionally removing duplicate lines"""

    captions = webvtt.read_buffer(io.StringIO(sub))
    dicts = [{"start": c.start, "end": c.end, "lines": c.lines} for c in captions]
    if dedupe:
        dicts = []
        prev_line = None
        for c in captions:
            if any("<c>" in l for l in c.lines):
                continue
            # Collect lines that are not dupes
            not_dupe_lines = []
            for line in c.lines:
                if not line.strip():
                    continue
                if line != prev_line:
                    not_dupe_lines.append(line)
                prev_line = line
            if not_dupe_lines:
                dicts.append({"start": c.start, "end": c.end, "lines": not_dupe_lines})
    if single:
        for d in dicts:
            d["line"] = "\n".join(d.pop("lines"))
    return dicts


def get_yt_meta(url, yt_metadata_args: dict) -> dict:
    """Return yt meta dict with meta data and/or subtitles
    yt_metadata_args is a dict of follwing format:
    yt_metadata_args = {
        'writesubtitles': True,
        'subtitleslangs': ['en'],
        'writeautomaticsub': True,
        'get_info': True
    }

    writesubtitles:    Whether to write subtitles
    writeautomaticsub: Write the automatically generated subtitles to a file
    subtitleslangs:    List of languages of the subtitles to download.
    get_info: whether to add info (title, description, tags etc) to the output.

    """

    write_subs = yt_metadata_args.get("writesubtitles", None)

    yt_metadata_args["skip_download"] = True
    yt_metadata_args["ignoreerrors"] = True
    yt_metadata_args["quiet"] = True

    info_dict, sub_dict = None, None

    with yt_dlp.YoutubeDL(yt_metadata_args) as yt:

        info_dict = yt.extract_info(url, download=False)
        if write_subs:
            sub_url = info_dict["requested_subtitles"][yt_metadata_args["subtitleslangs"][0]]["url"]
            res = requests.get(sub_url)
            sub = io.TextIOWrapper(io.BytesIO(res.content)).read()
            sub_dict = sub_to_dict(sub)

        if yt_metadata_args["get_info"]:
            info_dict.pop("subtitles")
            info_dict.pop("requested_formats")
            info_dict.pop("formats")
            info_dict.pop("thumbnails")
            info_dict.pop("automatic_captions")
        else:
            info_dict = None

        yt_meta_dict = {"info": info_dict, "subtitles": sub_dict}

        return yt_meta_dict


class Mp4Downloader:
    """Downloader class for mp4 links"""

    def __init__(self, timeout, tmp_dir):
        self.timeout = timeout
        self.tmp_dir = tmp_dir

    def __call__(self, url):
        resp = requests.get(url, stream=True, timeout=self.timeout)
        path = f"{self.tmp_dir}/{str(uuid.uuid4())}.mp4"
        with open(path, "wb") as f:
            f.write(resp.content)
        return path, None


class YtDlpDownloader:
    """Downloader class for yt-dlp links"""

    # TODO: maybe we just include height and width in the metadata_args
    def __init__(self, tmp_dir, metadata_args, video_height, video_width):
        self.tmp_dir = tmp_dir
        self.metadata_args = metadata_args
        self.video_height = video_height
        self.video_width = video_width

    def __call__(self, url):
        path = f"{self.tmp_dir}/{str(uuid.uuid4())}.mp4"
        format_string = (
            f"bv*[height<={self.video_height}][width<={self.video_width}][ext=mp4]"
            + f"+ba[ext=m4a]/b[height<={self.video_height}][width<={self.video_width}]"
        )
        ydl_opts = {
            "outtmpl": path,
            "format": format_string,
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)
        if self.metadata_args:
            yt_meta_dict = get_yt_meta(url, self.metadata_args)
        else:
            yt_meta_dict = None, None
        return path, yt_meta_dict, None


class VideoDataReader:
    """Video data reader provide data for a video"""

    def __init__(self, video_height, video_width, dl_timeout, tmp_dir, yt_meta_args) -> None:
        self.mp4_downloader = Mp4Downloader(dl_timeout, tmp_dir)
        self.yt_downloader = YtDlpDownloader(tmp_dir, yt_meta_args, video_height, video_width)

    def __call__(self, row):
        key, url = row

        yt_meta_dict = None
        # TODO: make nice function to detect what type of link we're dealing with
        if "youtube" in url:  # youtube link
            try:
                file_path, yt_meta_dict, error_message = self.yt_downloader(url)
            except Exception as e:  # pylint: disable=(broad-except)
                file_path, yt_meta_dict, error_message = None, None, str(e)
        # TODO: add .avi, .webm, should also work
        elif url.endswith(".mp4"):  # mp4 link
            file_path, error_message = self.mp4_downloader(url)
        else:
            file_path, error_message = None, "Warning: Unsupported URL type"

        if error_message is None:
            with open(file_path, "rb") as vid_file:
                vid_bytes = vid_file.read()
        else:
            vid_bytes = None

        if file_path is not None:  # manually remove tempfile
            os.remove(file_path)
        return key, vid_bytes, yt_meta_dict, error_message
