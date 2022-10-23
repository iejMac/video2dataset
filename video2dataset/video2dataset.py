"""Create dataset from video links and metadata."""


from .reader import Reader


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
  meta_refs = list(range(len(vids)))

  print(vids)
  print(ids)
  print(meta)



