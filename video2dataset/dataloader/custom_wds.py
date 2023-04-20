"""Custom WebDataset classes"""
import os
import numpy as np
import random
import tarfile
import warnings
import copy
from io import BufferedIOBase
from typing import Callable, Iterator, Tuple, cast, Optional, IO, Union, List

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import S3FileLoader, IterDataPipe
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple


import webdataset as wds
from webdataset import DataPipeline, filters, shardlists, cache, tariterators
from webdataset.compat import FluidInterface
from webdataset.autodecode import Decoder, ImageHandler


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [s[key] for s in samples] for key in keys}

    result = {}
    for key, values in batched.items():  # Iterate over both key and values
        first_value = values[0]
        if isinstance(first_value, (int, float)):
            if combine_scalars:
                result[key] = np.array(values)
        elif isinstance(first_value, torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(values)
        elif isinstance(first_value, np.ndarray):
            if combine_tensors:
                result[key] = np.array(values)
        else:
            result[key] = values

    return result


class KeyPassThroughDecoder(Decoder):
    """Decoder which allows you to pass through some keys"""

    def __init__(self, *args, passthrough_keys=None, **kwargs):
        """
        Initialize the KeyPassThroughDecoder.

        :param *args: Positional arguments to be passed to the base Decoder class.
        :param passthrough_keys: List of keys to bypass the decoding process.
        :param **kwargs: Keyword arguments to be passed to the base Decoder class.
        """
        super().__init__(*args, **kwargs)
        self.passthrough_keys = passthrough_keys or []  # Simplified passthrough_keys initialization

    def decode(self, sample):
        """
        Decode an entire sample.

        :param dict sample: The sample, a dictionary of key-value pairs.
        :return: Decoded sample.
        :rtype: dict
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in sample.items():  # Removed unnecessary list conversion
            if k[0] == "_":
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            else:
                assert (
                    isinstance(v, bytes) or k in self.passthrough_keys
                ), f"key: {k}; passthrough_keys: {self.passthrough_keys}"
                result[k] = self.decode1(k, v)
        return result


class FluidInterfaceWithChangedDecode(FluidInterface):
    def decode(
        self, *args, pre=None, post=None, only=None, partial=False, passthrough_keys=None, handler=wds.reraise_exception
    ):
        handlers = [ImageHandler(x) if isinstance(x, str) else x for x in args]
        decoder = KeyPassThroughDecoder(
            handlers, passthrough_keys=passthrough_keys, pre=pre, post=post, only=only, partial=partial
        )
        return self.map(decoder, handler=handler)


# TODO: pylint says this needs __getitem__
# pylint: disable=abstract-method
class WebDatasetWithChangedDecoder(DataPipeline, FluidInterfaceWithChangedDecode):
    """Small fluid-interface wrapper for DataPipeline."""

    def __init__(
        self,
        urls,
        handler=wds.reraise_exception,
        resampled=False,
        shardshuffle=None,
        cache_size=0,
        cache_dir=None,
        detshuffle=False,
        nodesplitter=shardlists.single_node_only,
        verbose=False,
    ):
        super().__init__()
        if isinstance(urls, IterableDataset):
            assert not resampled
            self.append(urls)
        elif isinstance(urls, dict):
            assert "datasets" in urls
            self.append(shardlists.MultiShardSample(urls))
        elif resampled:
            self.append(shardlists.ResampledShards(urls))
        else:
            self.append(shardlists.SimpleShardList(urls))
            self.append(nodesplitter)
            self.append(shardlists.split_by_worker)
            if shardshuffle is True:
                shardshuffle = 100
            if shardshuffle is not None:
                if detshuffle:
                    self.append(filters.detshuffle(shardshuffle))
                else:
                    self.append(filters.shuffle(shardshuffle))
        if cache_size == 0:
            self.append(tariterators.tarfile_to_samples(handler=handler))
        else:
            assert cache_size == -1 or cache_size > 0
            self.append(
                cache.cached_tarfile_to_samples(
                    handler=handler,
                    verbose=verbose,
                    cache_size=cache_size,
                    cache_dir=cache_dir,
                )
            )


def _s3dataset2samples(data, handler=wds.reraise_exception):
    for sample in data:
        try:
            # construct webdataset-style sample
            key = os.path.split(sample[0][0])[-1].split('.')[0]
            url = os.path.split(sample[0][0])[0]
            sample = {s[0].split('.')[-1]: s[1].read() for s in sample}
            sample["__key__"] = key
            sample["__url__"] = url

            yield sample
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


s3dataset2samples = filters.pipelinefilter(_s3dataset2samples)


class SeedSetter(IterDataPipe):
    """
    Resets the seed on call of __iter__ (invoked in the reset() method
    """
    def __init__(self, datapipe):
        super().__init__()
        self.datapipe = datapipe
        self.is_init = False
    # # def reset(self):
    def reset(self):
        # this will be called whenever __iter__ is invoked again (this should be kept in mind for shuffling
        if not self.is_init:
            # we only wanna do this once
            self.is_init = True

            worker_info = torch.utils.data.get_worker_info()

            if worker_info:
                worker_id = worker_info.id
                newseed = np.random.get_state()[1][0] + worker_id
                # print(f'New seed is {newseed}')
                np.random.seed(newseed)
                torch.random.manual_seed(newseed)
                random.seed(newseed)


    def __iter__(self)-> Iterator[Tuple[str, BufferedIOBase]]:
        # self.set_seed()
        # print(f'seed in worker init: {seed}')
        for data in self.datapipe:
            yield data

class PrefixResampler(IterDataPipe):

    def __init__(self, datapipe:IterDataPipe[str],prefixes:List[str],  ps:List[float] = None):
        super().__init__()
        urls = list(datapipe)
        self._len = len(urls)
        self.prefix2urls = {p: [] for p in set(prefixes)}
        self.ps = {k:p for k, p in zip(prefixes,ps)}
        if self.ps is None:
            # uniformly distributed
            self.ps = [1/len(self.prefix2urls)]*len(self.prefix2urls)


        print(f'{self.__class__.__name__} got the following prefixes: {prefixes}')
        for u in urls:
            self.prefix2urls[list(filter(lambda x: u.startswith(x),prefixes))[0]].append(u)




        for p in self.prefix2urls:
            if not self.prefix2urls[p]:
                print(f'removing prefix {p} from repefixes2urls since no_entries')
                self.prefix2urls.pop(p)
                self.ps.pop(p)

        sum_ = sum(list(self.ps.values()))
        self.ps = {k: self.ps[k] / sum_ for k in self.ps}

        print(f'Got the following (prob, prefix) pairs for {len(self.ps)} prefixes {[(k, p) for k, p in self.ps.items()]}')

        # internal iterator for one epoch
        self.it = 0
        self.url_pool = {}

        assert len(self.ps) == len(self.prefix2urls) and np.isclose(sum(self.ps.values()),1.), 'Probabilities must have the same length than prefix and must sum up to 1'

    def reset(self):
        # this will be called whenever __iter__ is invoked again (this should be kept in mind for shuffling
        print("refilling url_pool")
        self.url_pool = copy.deepcopy(self.prefix2urls)
        self.it=0

    def refill_prefix(self, prefix):
        # refill the buffer
        self.url_pool[prefix] = copy.deepcopy(self.prefix2urls[prefix])


    def __iter__(self):
        while self.it < self.__len__():

            # sample prefix with corresponding probs
            prefix_id = np.random.choice(len(self.ps),1,p=list(self.ps.values())).item()
            prefix = list(self.ps.keys())[prefix_id]
            # refill the url pool for the selected prefix if empty
            if not self.url_pool[prefix]:
                self.refill_prefix(prefix)

            # uniformly sample from all available urls for the selected prefix
            url_id = np.random.randint(len(self.url_pool[prefix]), dtype=int)
            url = self.url_pool[prefix].pop(url_id)

            yield url
            self.it += 1


    def __len__(self):
        return self._len



class TarArchiveLoaderAndCloser(TarArchiveLoader):

    def __init__(self, handler:Callable = wds.reraise_exception, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handler = handler
    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(data_stream.file_obj, tarfile.TarFile):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (
                        self.mode
                        if hasattr(data_stream, "seekable") and data_stream.seekable()
                        else self.mode.replace(":", "|")
                    )
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=reading_mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(f"failed to extract file {tarinfo.name} from source tarfile {pathname}")
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)  # type: ignore[misc]

                # close tarfile after it's been exceeded
                tar.close()
                del tar
                # if isinstance(data_stream, StreamWrapper):
                #     data_stream.autoclose()
            except Exception as e:
                warnings.warn(f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!")
                if self.handler(e):
                    if hasattr(e, "args") and len(e.args) > 0:
                        e.args = (e.args[0] + " @ " + str(extracted_fobj),) + e.args[1:]
                else:
                    raise e
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()
                del data_stream


def grouper(x):
    return x[0].split("/")[-1].split(".")[0]


class S3TorchDataWebdataset(DataPipeline,FluidInterfaceWithChangedDecode):

    def __init__(self,urls:Union[List[str], str],
                 repeat:int=None,
                 shardshuffle:int=10000,
                 sample_shuffle:int=0,
                 buffer_size:int=None,
                 resample_prefixes:bool=False,
                 prefix_probs:Optional[List[float]]=None,
                 handler:Callable=wds.reraise_exception):
        super().__init__()

        if isinstance(urls, (List, list)):
            pass

        elif isinstance(urls, str):
            urls = [urls]
        else:
            raise TypeError('urls need to be path to a S3 prefix or list of paths to more than one prefixes')

        # sharding filter ensures propper splitting for distributed environment
        s3_datapipe = (IterableWrapper(urls)
                       .list_files_by_s3(masks="**/*.tar")
                       .sharding_filter())

        # after this operation datapipes in the distinct processes contain different tars
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        s3_datapipe.apply_sharding(world_size,global_rank)
        # synchronize data across processes to prevent hanging if sharding is uneven (which is likely)
        s3_datapipe = s3_datapipe.fullsync()

        # start shuffling accross shards for the first time to mix different datasets
        # (can be the same for all workers, just as an additional shuffled initialization)
        if shardshuffle>1 and not resample_prefixes:
            raw_tars = list(s3_datapipe)
            random.shuffle(raw_tars)
            print('Loader got the following concrete shards (first 25 are shown)')
            print(raw_tars[:25])
            # back to datapipes
            s3_datapipe = IterableWrapper(raw_tars)
        elif resample_prefixes:
            s3_datapipe = PrefixResampler(s3_datapipe, prefixes=urls, ps=prefix_probs)
        s3_datapipe = SeedSetter(s3_datapipe)
        s3_datapipe = (s3_datapipe
                       .shuffle(buffer_size=shardshuffle)
                       .cycle(count=repeat))

        # s3_datapipe = s3_datapipe.sharding_filter()

        # Load data with S3FileLoader
        s3_datapipe = S3FileLoader(s3_datapipe, buffer_size=buffer_size)
        # adapted TarLoader which closes open tarfile handles after exceeding them
        s3_datapipe = (TarArchiveLoaderAndCloser(datapipe=s3_datapipe, handler=handler).
                       groupby(grouper))
        if sample_shuffle > 0:
            s3_datapipe = s3_datapipe.shuffle(buffer_size=sample_shuffle)

        self.append(s3_datapipe)
        self.append(s3dataset2samples(handler=handler))