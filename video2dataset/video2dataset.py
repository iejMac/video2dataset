"""Create dataset from video links and metadata."""


from .reader import Reader
from .writer import FileWriter, WebDatasetWriter
from .downloader import handle_url
from tqdm import tqdm


def video2dataset(
    src,
    dest="",
    output_format="webdataset",
    metadata_columns="",
    video_format="",
    sample_rate=None,
    url_col='videoLoc'
):
    """
    create video dataset from video links

    src:
      str: path to table of data with video links and metdata
    dest:
      str: where to save dataset to
    output_format:
      str: webdataset, files
    metadata_columns:
      str: a comma separated list of metadata column names to look for in src
    """
    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns] if metadata_columns != "" else []
    metadata_columns = list(metadata_columns) if isinstance(
        metadata_columns, tuple) else metadata_columns
    reader = Reader(src, metadata_columns, url_col)
    vids, ids, meta = reader.get_data()
    video_format = video_format.replace(' ', '')

    starting_shard_id = 0
    shard_sample_count = 10000

    if output_format == "files":
        writer = FileWriter(dest, video_format)
    elif output_format == "webdataset":
        writer = WebDatasetWriter(
            dest, 9, video_format=video_format, maxcount=shard_sample_count, shard_id=starting_shard_id)

    for i in tqdm(range(len(vids))):
        vid = vids[i]
        vid_id = ids[i]
        vid_meta = {}
        for k in meta:
            vid_meta[k] = meta[k][i].as_py()

        # NOTE: Right now assuming video is url (maybe add support for local mp4
        streams = handle_url(
            vid, video_format=video_format, sample_rate=sample_rate)
        info = streams['info']

        vid_meta = {
            'meta': vid_meta,
            'video_info': info,
            'error': streams['error']
        }

        writer.write(streams, vid_id, vid_meta)
