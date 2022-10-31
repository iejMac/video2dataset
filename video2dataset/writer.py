"""save embeddings."""
import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO


class FileWriter:
    """Writes output as files."""

    def __init__(self, output_folder):
        self.output_folder = output_folder

        self.fs, self.output_folder = fsspec.core.url_to_fs(output_folder)

    def write(self, video, key, metadata=None):
        """write sample to file."""
        key = str(key)

        save_pth = os.path.join(self.output_folder, key + ".mp4")
        with self.fs.open(save_pth, "wb") as f:
            f.write(video)

        if metadata is not None:
            if "caption" in metadata:
                caption = str(metadata.pop("caption"))
                caption_filename = os.path.join(self.output_folder, key + ".txt")
                with self.fs.open(caption_filename, "w") as f:
                    f.write(caption)
            if len(metadata) > 0:
                j = json.dumps(metadata, indent=4)
                meta_filename = os.path.join(self.output_folder, key + ".json")
                with self.fs.open(meta_filename, "w") as f:
                    f.write(j)

    def close(self):
        pass
