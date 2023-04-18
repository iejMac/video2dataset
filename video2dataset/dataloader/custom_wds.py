"""Custom WebDataset classes"""
import os
import re
import numpy as np
import torch
import tarfile
import warnings
import math
from torch.utils.data import IterableDataset
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import S3FileLoader
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from io import BufferedIOBase
from typing import Callable, Iterator, Tuple, cast, Optional, IO, Union, List



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


# def url_opener_(data, handler=wds.warn_and_continue, **kw):
#     """Given a stream of url names (packaged in `dict(url=url)`), yield opened streams."""
#     for sample in data:
#         assert isinstance(sample, dict), sample
#         assert "url" in sample
#         url = sample["url"]
#         try:
#             stream = gopen(url, **kw)
#             sample.update(stream=stream)
#             yield sample
#         except Exception as exn:
#             exn.args = exn.args + (url,)
#             if handler(exn):
#                 continue
#             else:
#                 break
#
#
# def tar_file_iterator_return_all(fileobj, skip_meta=r"__[^/]*__($|/)", handler=wds.warn_and_stop):
#     """Iterate over tar file, yielding filename, content pairs for the given tar stream.
#
#     :param fileobj: byte stream suitable for tarfile
#     :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")
#
#     """
#     stream = tarfile.open(fileobj=fileobj, mode="r|*")
#     for tarinfo in stream:
#         fname = tarinfo.name
#         corrupted = False
#         load_data = True
#         try:
#             if not tarinfo.isreg():
#                 data = None
#                 load_data = False
#                 corrupted = True
#             if fname is None:
#                 # fname = f'dummy/{dummy_name_id}/{dummy_in_sample_id}'
#                 corrupted = True
#
#             if (
#                 "/" not in fname
#                 and fname.startswith(tariterators.meta_prefix)
#                 and fname.endswith(tariterators.meta_suffix)
#             ):
#                 # skipping metadata for now
#                 continue
#             if skip_meta is not None and re.match(skip_meta, fname):
#                 continue
#             if load_data:
#                 data = stream.extractfile(tarinfo).read()
#             result = dict(fname=fname, data=data, corrupted=corrupted)
#             yield result
#             stream.members = []
#         except Exception as exn:
#             if hasattr(exn, "args") and len(exn.args) > 0:
#                 exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
#             if handler(exn):
#                 continue
#             else:
#                 result = dict(fname=fname,data=None, corrupted=True)
#                 yield result
#     del stream
#
#
# def tar_file_expander_return_all(data, handler=wds.warn_and_continue):
#     """Expand a stream of open tar files into a stream of tar file contents.
#
#     This returns an iterator over (filename, file_contents).
#     """
#     for source in data:
#         url = source["url"]
#         try:
#             assert isinstance(source, dict)
#             assert "stream" in source
#             for sample in tar_file_iterator_return_all(source["stream"], handler=wds.warn_and_stop):
#                 assert (
#                     isinstance(sample, dict) and "data" in sample and "fname" in sample and 'corrupted' in sample
#                 )
#                 sample["__url__"] = url
#                 yield sample
#         except Exception as exn:
#             exn.args = exn.args + (source.get("stream"), source.get("url"))
#             if handler(exn):
#                 continue
#             else:
#                 break
#
# def base_plus_ext(path):
#     """Split off all file extensions.
#
#     Returns base, allext.
#
#     :param path: path with extensions
#     :param returns: path with all extensions removed
#
#     """
#     match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", str(path))
#     if not match:
#         return None, None
#     return match.group(1), match.group(2)
#
# def get_running_mean(subset:List, current_val:int, max_subset_length:int=10000):
#     if len(subset) >= max_subset_length:
#         subset = subset[1:]
#     subset.append(current_val)
#     count = len(subset)
#     return sum(subset)/count
#
# #TODO test
# def group_by_keys_return_all(data, suffixes, keys=base_plus_ext,
#                              lcase=True, handler=wds.warn_and_stop):
#     """Return function over iterator that groups key, value pairs into samples.
#
#     :param keys: function that splits the key into key and extension (base_plus_ext)
#     :param lcase: convert suffixes to lower case (Default value = True)
#     """
#     current_sample = None
#     # in case the first field of a new sampe has no prefix
#     # this is to track the mean
#     fields_without_prefix = 0
#     is_sample_corrupted = False
#     for filesample in data:
#         try:
#             assert isinstance(filesample, dict)
#             fname, value, corrupted = filesample["fname"], filesample["data"], filesample['corrupted']
#             prefix, suffix = keys(fname)
#             is_sample_corrupted |= corrupted
#
#             fields_without_prefix +=int(prefix is None)
#
#             if prefix is None:
#                 # current_sample = dict(__key__='dummy-corrupted', __url__=filesample['__url__'], __corrupted__=True)
#                 assert corrupted
#
#
#             if suffix and lcase:
#                 suffix = suffix.lower()
#
#             if current_sample is None or prefix != current_sample["__key__"]:
#                 if tariterators.valid_sample(current_sample) and \
#                             len(current_sample) + fields_without_prefix >= len(suffixes)+2:
#
#                     for key in set(suffixes).difference(current_sample):
#                         current_sample[key] = None
#
#                     current_sample['__corrupted__'] = is_sample_corrupted
#                     fields_without_prefix = 0
#                     is_sample_corrupted = False
#                     yield current_sample
#                     current_sample = None
#                 if prefix is not None:
#                     current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
#
#
#             if suffix in current_sample:
#                 raise ValueError(
#                     f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
#                 )
#             if suffix in suffixes:
#                 current_sample[suffix] = value
#         except Exception as exn:
#             exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
#             if handler(exn):
#                 continue
#             else:
#                 break
#     if tariterators.valid_sample(current_sample):
#         yield current_sample
#
#
# def tarfile_samples_return_all(src, suffixes, handler=wds.warn_and_stop):
#     streams = url_opener_(src)
#     files = tar_file_expander_return_all(streams, handler=handler)
#     samples = group_by_keys_return_all(files, suffixes=suffixes, handler=handler)
#     return samples
#
# tarfile_to_samples_return_all = filters.pipelinefilter(tarfile_samples_return_all)
#
#
# def add_corrupted(sample):
#     result = sample
#     if isinstance(result,dict):
#         result['__corrupted__'] = True
#     elif isinstance(result, (tuple, list)):
#         result += ({'__corrupted__': True})
#     else:
#         # should only happen after decoding functions
#         result = '__corrupted__'
#
#     sample = result
#     return sample
# def _return_all_map(data, f, handler=wds.warn_and_stop):
#     """Map samples."""
#     for sample in data:
#         try:
#             result = f(sample)
#         except Exception as exn:
#             if handler(exn):
#                 continue
#             else:
#                 result = add_corrupted(sample)
#         if result is None:
#             result = add_corrupted(sample)
#         if isinstance(sample, dict) and isinstance(result, dict):
#             result["__key__"] = sample.get("__key__")
#             if '__corrupted__' not in result:
#                 result['__corrupted__'] = False
#         yield result
#
# return_all_map = wds.pipelinefilter(_return_all_map)
#
#
# def _map_dict_return_all(data, handler=wds.warn_and_stop, **kw):
#     """Map the entries in a dict sample with individual functions."""
#     assert len(list(kw.keys())) > 0
#     for key, f in kw.items():
#         assert callable(f), (key, f)
#
#     for sample in data:
#         assert isinstance(sample, dict)
#         try:
#             for k, f in kw.items():
#                 sample[k] = f(sample[k])
#         except Exception as exn:
#             if handler(exn):
#                 continue
#             else:
#                 sample = add_corrupted(sample)
#         yield sample
#
#
# map_dict_return_all = wds.pipelinefilter(_map_dict_return_all)
#
#
# def _to_tuple_return_all(
#     data, *args, handler=wds.warn_and_stop, missing_is_error=True, none_is_error=None
# ):
#     """Convert dict samples to tuples."""
#     if none_is_error is None:
#         none_is_error = missing_is_error
#     if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
#         args = args[0].split()
#
#     for sample in data:
#         try:
#             result = tuple(
#                 [filters.getfirst(sample, f, missing_is_error=missing_is_error) for f in args]
#             )
#             if none_is_error and any(x is None for x in result):
#                 raise ValueError(f"to_tuple {args} got {sample.keys()}")
#             # if we get to this point we should be healthy except when outputs are none
#             result += ({'__corrupted__': any(x is None for x in result)})
#             yield result
#         except Exception as exn:
#             if handler(exn):
#                 continue
#             else:
#                 result = tuple(
#                     [filters.getfirst(sample, lambda x: x, missing_is_error=missing_is_error) for f in args]+ [{'__corrupted__': True}]
#                 )
#                 yield result
#
#
# to_tuple_return_all = wds.pipelinefilter(_to_tuple_return_all)
#
#
# class ReturnAllWebDataset(wds.WebDataset):
#
#
#     def pop(self,id_=-1):
#         self.pipeline.pop(id_)
#     def __init__(self, suffixes:List[str], *args, **kwargs):
#         super().__init__(*args,**kwargs)
#         self.pop()
#
#         self.append(tarfile_to_samples_return_all(suffixes=suffixes,
#                                                   handler=wds.warn_and_stop))
#
#     def map(self, f, handler=wds.warn_and_stop):
#         return self.compose(return_all_map(f, handler=handler))
#
#     def map_tuple(self, *args, handler=wds.warn_and_stop):
#         return self.compose(to_tuple_return_all(*args, handler=handler))
#     def map_dict(self, handler=wds.warn_and_stop, **kw):
#         return self.compose(map_dict_return_all(handler = handler,**kw))
#
# class AddDefaultFields(object):
#
#
#     default_dict = {}
#
#     def __init__(self, to_return:List[str], skip:bool=True):
#         self.to_return = set(to_return)
#         assert all([x in self.default_dict for x in self.to_return]), f'missing keys in in default dict: {self.to_return.difference(self.default_dict)}'
#         self.skip = skip
#
#
#     def __call__(self, x):
#         if self.skip:
#             return x
#
#
#         for key in self.to_return.intersection(x):
#             x[key] = self.default_dict[key]
#
#         return x



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


class S3TorchDataWebdataset(DataPipeline,FluidInterface):

    def __init__(self,urls:Union[List[str], str],
                 repeat:int=None,
                 shardshuffle:int=10000,
                 sample_shuffle:int=0,
                 buffer_size:int=None,
                 handler=wds.reraise_exception):
        super().__init__()

        if isinstance(urls, (List, list)):
            pass

        elif isinstance(urls, str):
            urls = [urls]
        else:
            raise TypeError('urls need to be path to a S3 prefix or list of paths to more than one prefixes')

        # for url in urls:



        s3_datapipe = IterableWrapper(urls).list_files_by_s3(masks="**/*.tar")
        s3_datapipe = s3_datapipe.shuffle(buffer_size=shardshuffle).cycle(count=repeat)


        # ensure propper splitting for distributed environment
        s3_datapipe = s3_datapipe.sharding_filter()

        # Load data with S3FileLoader
        s3_datapipe = S3FileLoader(s3_datapipe, buffer_size=buffer_size)
        # adapted TarLoader which closes open tarfile handles after exceeding them
        s3_datapipe = (TarArchiveLoaderAndCloser(datapipe=s3_datapipe, handler=handler).
                       groupby(grouper))
        if sample_shuffle > 0:
            s3_datapipe = s3_datapipe.shuffle(buffer_size=sample_shuffle)

        self.append(s3_datapipe)
        self.append(s3dataset2samples(handler=handler))

