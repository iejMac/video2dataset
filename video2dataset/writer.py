"""save embeddings."""
import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO


class FileWriter:
    """Writes output as files."""

    def __init__(self, output_folder, video_format):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.output_folder = output_folder

        self.fs, self.output_folder = fsspec.core.url_to_fs(output_folder)
        self.video_format = video_format

    def write(self, streams, key, metadata=None):
        """write sample to file."""
        key = str(key)

        for ext in self.video_format.split(','):
            try:
                video = streams[ext]['file']
                save_pth = os.path.join(self.output_folder, key + '.' + ext)
                with self.fs.open(save_pth, "wb") as f:
                    f.write(video)
            except:
                continue
        if metadata is not None:
            if "caption" in metadata:
                caption = str(metadata.pop("caption"))
                caption_filename = os.path.join(
                    self.output_folder, key + ".txt")
                with self.fs.open(caption_filename, "w") as f:
                    f.write(caption)
            if len(metadata) > 0:
                j = json.dumps(metadata, indent=4)
                meta_filename = os.path.join(self.output_folder, key + ".json")
                with self.fs.open(meta_filename, "w") as f:
                    f.write(j)

    def close(self):
        pass


class WebDatasetWriter:
    """Writes output in WebDataset format."""

    def __init__(
            self,
            output_folder: str,
            oom_shard_count: int,
            video_format: str,
            maxcount: int = 10000,
            shard_id: int = 0
    ):
        self.output_folder = output_folder
        self.oom_shard_count = oom_shard_count
        self.video_format = video_format
        self.maxcount = maxcount
        self.shard_id = shard_id

        self.count = 0

        self.tarwriter = None
        self.tar_fd = None

        self.create_shard()

    def create_shard(self):
        """create new shard in sequential order."""
        self.close()
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=self.shard_id, oom_shard_count=self.oom_shard_count
        )
        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)

    def write(self, streams, key, metadata=None):
        """write sample to current shard."""
        key = str(key)
        if self.count >= self.maxcount:
            self.shard_id += 1
            self.count = 0
            self.create_shard()

        for encode_format in self.video_format.split(','):

            try:
                sample = {"__key__": key,
                          encode_format: streams[encode_format]['file']}
            except:
                sample = {"__key__": key,
                          encode_format: streams['error']}

        if metadata is not None:
            if "caption" in metadata:
                sample["txt"] = str(metadata.pop("caption"))
            if len(metadata) > 0:
                sample["json"] = json.dumps(metadata, indent=4)

        self.tarwriter.write(sample)
        self.count += 1

    def close(self):
        if self.tarwriter is not None:
            self.tarwriter.close()
            self.tar_fd.close()
