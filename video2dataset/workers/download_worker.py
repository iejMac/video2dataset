"""the downloader module handles the downloading"""
import fsspec
import math
from multiprocessing.pool import ThreadPool
import pyarrow as pa
from threading import Semaphore
import time
import traceback
from typing import cast

from video2dataset.data_reader import VideoDataReader
from video2dataset.logger import write_stats
from video2dataset.workers.worker import ShardStatus, Streams, get_subsamplers, process_sample


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class DownloadWorker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        save_caption,
        output_folder,
        column_list,
        tmp_dir,
        encode_formats,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = save_caption
        self.output_folder = output_folder
        self.column_list = column_list
        self.input_encode_formats = encode_formats
        self.config = config
        self.data_reader = VideoDataReader(encode_formats, tmp_dir, config["reading"])
        self.url_indice = self.column_list.index("url")
        self.caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        self.oom_sample_per_shard = math.ceil(math.log10(self.config["storage"]["number_sample_per_shard"]))
        self.subsamplers, self.output_encode_formats = get_subsamplers(
            config,
            encode_formats,
            do_clipping=("clips" in self.column_list),
        )

    def __call__(
        self,
        row,
    ):
        try:
            shard_file, shard_id = row
            self.process_shard(shard_file, shard_id)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def get_shard_processors(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Get objects for loading and writing data"""

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        print(shard_path)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
            schema = df.schema
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
        )
        shard_sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.config["storage"]["oom_shard_count"],
            schema,
            self.output_encode_formats,
        )
        pydict = df.select(self.column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))

        def rm_shard_path():
            fs.rm(shard_path)

        return shard_sample_writer, shard_to_dl, rm_shard_path

    def process_shard(
        self,
        shard_file: str,
        shard_id: int,
    ):
        """Function to start an video downloading in one process"""

        start_time = time.time()
        shard_sample_writer, shard_to_dl, rm_shard_path = self.get_shard_processors(shard_file, shard_id)
        shard_status = ShardStatus(count=len(shard_to_dl))

        semaphore = Semaphore(self.config["distribution"]["thread_count"])
        def data_generator():
            for key_and_url in [(key, x[self.url_indice]) for key, x in shard_to_dl]:
                semaphore.acquire()  # pylint: disable=consider-using-with
                yield key_and_url

        data_reader_call_param_generator = data_generator()

        with ThreadPool(self.config["distribution"]["thread_count"]) as thread_pool:
            for key, streams, yt_meta_dict, shard_status.error_message in thread_pool.imap_unordered(
                self.data_reader,  # pylint: disable=(unnecessary-lambda)
                data_reader_call_param_generator,
            ):
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(
                        key, shard_id, self.oom_sample_per_shard, self.config["storage"]["oom_shard_count"]
                    )
                    caption = sample_data[self.caption_indice] if self.caption_indice is not None else None
                    metadata = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": shard_status.error_message,
                        "yt_meta_dict": yt_meta_dict,
                    }
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                    semaphore.release()
                    return

                try:
                    if shard_status.error_message is not None:
                        print(shard_status.error_message)
                        if "[youtube]" in shard_status.error_message:  # video-specific error, remove videoID
                            shard_status.error_message = "ERROR: [youtube]:" + shard_status.error_message.split(":")[-1]
                        raise ValueError
                except Exception:  # pylint: disable=broad-except
                    shard_status.failed["failed_to_download"] += 1
                    shard_status.status_dict.increment(shard_status.error_message)
                    metadata["status"] = "failed_to_download"
                    metadata["error_message"] = shard_status.error_message
                    shard_sample_writer.write(
                        {},
                        str_key,
                        sample_data[self.caption_indice] if self.caption_indice is not None else None,
                        metadata,
                    )
                    semaphore.release()
                    return

                for stream in streams.values():
                    shard_status.bytes_downloaded += len(stream)
                for modality in streams:
                    streams[modality] = [streams[modality]]

                process_sample(
                    subsamplers=self.subsamplers,
                    shard_status=shard_status,
                    streams=cast(Streams, streams),
                    key=str_key,
                    caption=cast(str, caption),
                    metadata=metadata,
                    captions_are_subtitles=self.config["storage"]["captions_are_subtitles"],
                    shard_sample_writer=shard_sample_writer,
                )
                semaphore.release()

            shard_sample_writer.close()
        rm_shard_path()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            shard_status.count,
            shard_status.successes,
            shard_status.failed["failed_to_download"],
            shard_status.failed["failed_to_subsample"],
            shard_status.bytes_downloaded,
            start_time,
            end_time,
            shard_status.status_dict,
            self.config["storage"]["oom_shard_count"],
        )
