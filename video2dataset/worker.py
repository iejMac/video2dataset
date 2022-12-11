"""the downloader module handles the downloading"""

import math
import time
import pyarrow as pa
import traceback

import fsspec

from multiprocessing.pool import ThreadPool
from threading import Semaphore

from video2dataset.data_reader import VideoDataReader
from .logger import CappedCounter
from .logger import write_stats
from .subsampler import NoOpSubsampler, ClippingSubsampler


def compute_key(key, shard_id, oom_sample_per_shard, oom_shard_count):
    true_key = (10**oom_sample_per_shard) * shard_id + key
    key_format = oom_sample_per_shard + oom_shard_count
    str_key = "{true_key:0{key_format}d}".format(  # pylint: disable=consider-using-f-string
        key_format=key_format, true_key=true_key
    )
    return str_key


class Worker:
    """The downloader class gets calls with shards, download them then call the writer to write them down"""

    def __init__(
        self,
        sample_writer_class,
        save_caption,
        output_folder,
        column_list,
        thread_count,
        timeout,
        find_format_timeout,
        number_sample_per_shard,
        oom_shard_count,
        encode_format,
        max_format_tries,
        video_height,
        video_width,
        oom_clip_count=5,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.save_caption = save_caption
        self.output_folder = output_folder
        self.column_list = column_list
        self.number_sample_per_shard = number_sample_per_shard
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        self.thread_count = thread_count
        self.data_reader = VideoDataReader(video_height, video_width, timeout, find_format_timeout, max_format_tries)
        self.noop_subsampler = NoOpSubsampler()
        self.clipping_subsampler = ClippingSubsampler(oom_clip_count)

    def __call__(
        self,
        row,
    ):
        try:
            self.download_shard(row)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def download_shard(
        self,
        row,
    ):
        """Function to start an video downloading in one process"""

        shard_id, shard_file = row
        start_time = time.time()

        fs, shard_path = fsspec.core.url_to_fs(shard_file)
        with fs.open(shard_path, "rb") as f:
            df = pa.ipc.open_file(f).read_all()
        schema = df.schema
        schema = (
            schema.append(pa.field("key", pa.string()))
            .append(pa.field("status", pa.string()))
            .append(pa.field("error_message", pa.string()))
        )

        pydict = df.select(self.column_list).to_pydict()
        shard_to_dl = list(enumerate(zip(*(pydict[col] for col in self.column_list))))
        del pydict
        del df

        status_dict = CappedCounter()

        count = len(shard_to_dl)
        successes = 0
        failed_to_download = 0
        failed_to_subsample = 0
        url_indice = self.column_list.index("url")
        caption_indice = self.column_list.index("caption") if "caption" in self.column_list else None
        key_url_list = [(key, x[url_indice]) for key, x in shard_to_dl]

        semaphore = Semaphore(self.thread_count)

        def data_generator():
            for e in key_url_list:
                semaphore.acquire()  #  pylint: disable=(consider-using-with)
                yield e

        loader = data_generator()

        # give schema to writer
        sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            self.save_caption,
            self.oom_shard_count,
            schema,
            self.encode_format,
        )
        oom_sample_per_shard = math.ceil(math.log10(self.number_sample_per_shard))

        with ThreadPool(self.thread_count) as thread_pool:
            for key, vid_stream, error_message in thread_pool.imap_unordered(
                lambda x: self.data_reader(x),  #  pylint: disable=(unnecessary-lambda)
                loader,
            ):
                # print(semaphore._Semaphore__value)
                try:
                    _, sample_data = shard_to_dl[key]
                    str_key = compute_key(key, shard_id, oom_sample_per_shard, self.oom_shard_count)
                    meta = {
                        **{self.column_list[i]: sample_data[i] for i in range(len(self.column_list))},
                        "key": str_key,
                        "status": None,
                        "error_message": error_message,
                    }

                    if error_message is not None:
                        failed_to_download += 1
                        status = "failed_to_download"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        '''
                        sample_writer.write(
                            None,
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        '''
                        semaphore.release()
                        continue

                    tmpdir = None
                    '''
                    if "clips" in self.column_list:
                        tmpdir, subsampled_videos, metas, error_message = self.clipping_subsampler(vid_stream, meta)
                    else:
                        subsampled_videos, metas, error_message = self.noop_subsampler(vid_stream, meta)
                    '''

                    if error_message is not None:
                        failed_to_subsample += 1
                        status = "failed_to_subsample"
                        status_dict.increment(error_message)
                        meta["status"] = status
                        meta["clips"] = []
                        meta["error_message"] = error_message
                        '''
                        sample_writer.write(
                            None,
                            str_key,
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                        '''
                        semaphore.release()
                        if tmpdir is not None:
                            tmpdir.cleanup()
                        continue

                    successes += 1
                    status = "success"
                    status_dict.increment(status)
                    '''
                    for subsampled_video, meta in zip(subsampled_videos, metas):
                        meta["status"] = status
                        sample_writer.write(
                            subsampled_video,
                            meta["key"],
                            sample_data[caption_indice] if caption_indice is not None else None,
                            meta,
                        )
                    '''
                    if tmpdir is not None:
                        tmpdir.cleanup()
                except Exception as err:  # pylint: disable=broad-except
                    traceback.print_exc()
                    print(f"Sample {key} failed to download: {err}")
                semaphore.release()

            sample_writer.close()
            thread_pool.terminate()
            thread_pool.join()
            del thread_pool

        end_time = time.time()
        write_stats(
            self.output_folder,
            shard_id,
            count,
            successes,
            failed_to_download,
            failed_to_subsample,
            start_time,
            end_time,
            status_dict,
            self.oom_shard_count,
        )
        fs.rm(shard_path)
