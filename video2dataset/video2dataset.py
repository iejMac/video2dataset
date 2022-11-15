"""Create dataset from video links and metadata."""


from .reader import Reader
from .writer import FileWriter, WebDatasetWriter
from .downloader import handle_url
from .downloader import handle_mp4_link


def video2dataset(
  src,
  dest="",
  output_format="webdataset",
  metadata_columns="",
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
  metadata_columns = list(metadata_columns) if isinstance(metadata_columns, tuple) else metadata_columns
  reader = Reader(src, metadata_columns)
  vids, ids, meta = reader.get_data()

  starting_shard_id = 0
  shard_sample_count = 10000

  if output_format == "files":
    writer = FileWriter(dest)
  elif output_format == "webdataset":
    writer = WebDatasetWriter(dest, 9, "mp4", maxcount=shard_sample_count, shard_id=starting_shard_id)

  for i in range(len(vids)):
    print(f"{i}/{len(vids)}")
    vid = vids[i]
    vid_id = ids[i]
    vid_meta = {}
    for k in meta:
      vid_meta[k] = meta[k][i].as_py()

    # NOTE: Right now assuming video is url (maybe add support for local mp4
    load_vid, file, dst_name = handle_url(vid)

    with open(load_vid, "rb") as vid_file:
      vid_bytes = vid_file.read()
    video = vid_bytes

    writer.write(video, vid_id, vid_meta)

    if file is not None:  # for python files that need to be closed
      file.close()
