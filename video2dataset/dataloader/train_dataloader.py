"""video dataset creation"""
import logging
import math
import random
import webdataset as wds
from functools import partial

from dataclasses import dataclass
from multiprocessing import Value
from torch.utils.data import DataLoader
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from .custom_wds import WebDatasetWithChangedDecoder, dict_collation_fn
from .transform import VideoResizer, CutsAdder
from .video_decode import VideoDecorder, VideoDecorderWithCutDetection
from .filters import *


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    """info about data"""

    dataloader: DataLoader
    shared_epoch: SharedEpoch
    sampler = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def filter_no_caption_or_no_video(sample):
    has_caption = "txt" in sample
    has_video = "mp4" in sample
    return has_caption and has_video


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):  # pylint: disable=unused-argument
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample  # pylint: disable=unsupported-membership-test
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class detshuffle2(wds.PipelineStage):  # pylint: disable=invalid-name, abstract-method
    """."""

    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        """run"""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def reassemble(x):
    new_dict = dict()

    for key in x:
        if key not in f"mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            continue

        # this is updating the output of video decoders
        if isinstance(x[key], tuple) and len(x[key]) == 2:
            new_dict.update({f'{subk}': x[key][-1][subk] for subk in x[key][-1]})

        x[key] = x[key][0]
    x.update(new_dict)
    del new_dict
    return x

def exists(cls, attr):
    if isinstance(cls,dict):
        return attr in cls
    else:
        return hasattr(cls,attr)


def get_video_dataset(args,):

    # TODO think about different ways of pasing args to dataloader (personally, I prefer omegaconf)
    tars = args.data
    # batching
    batch_size = args.batch_size
    drop_last = args.drop_last if exists(args,'drop_last') else False
    # basic webdataset params
    shardshuffle=args.shardshuffle if exists(args,'shardshuffle') else None
    cuts_key = args.cuts_key if exists(args,'cut_key') else None
    shuffle = args.shuffle if exists(args,'shuffle') else False
    repeat = args.repeat if exists(args,'repeat') else False
    video_key = args.video_key if exists(args,'video_key') else 'mp4'
    decoder_kwargs = args.decoder_kwargs if exists(args,'decoder_kwargs') else dict()
    # asethetics filter
    aesthetics_threshold = args.aesthetics_threshold if exists(args,'aesthetics_threshold') else None
    # language_filter
    allowed_languages = args.allowed_languages if exists(args,'allowed_languages') else None
    # p_unsafe filter
    p_unsafe_threshold = args.p_unsafe_threshold if exists(args, 'p_unsafe_threshold') else None
    # resizer params
    resize_size = args.resize_size if exists(args, 'resize_size') else None
    crop_size = args.crop_size if exists(args, 'crop_size') else None
    # default center crop
    random_crop = args.random_crop if exists(args, 'random_crop') else False
    # get original height and width from dataset
    original_height_key = args.original_height_key if exists(args, 'original_height_key') else 'original_height'
    original_width_key = args.original_width_key if exists(args, 'original_width_key') else 'original_width'


    additional_decoder_kwargs = {}
    if cuts_key:
        dataset_cls = WebDatasetWithChangedDecoder
        video_decoder_cls = partial(VideoDecorderWithCutDetection, cuts_key=cuts_key)
        additional_decoder_kwargs = {'passthrough_keys': [video_key]}
    else:
        dataset_cls = wds.WebDataset
        video_decoder_cls = VideoDecorder


    dset = dataset_cls(tars,nodesplitter=wds.split_by_node,shardshuffle=shardshuffle,handler=wds.warn_and_continue)

    if repeat:
        dset = dset.repeat()

    if shuffle:
        dset = dset.shuffle(shuffle)


    key_filter = KeyFilter(video_key=video_key)
    dset = dset.select(key_filter)

    # add cut detection
    if cuts_key:
        cut_adder = CutsAdder(cuts_key=cuts_key, video_key=video_key)
        dset = dset.map(cut_adder, handler=wds.warn_and_continue)

    # various optional filters TODO add more if needed
    aesthetics_filter = AestheticsFilter(aesthetic_thld=aesthetics_threshold)
    language_filter = LanguageFilter(languages=allowed_languages)
    unsafe_filter = UnsafeFilter(p_unsafe_threshold=p_unsafe_threshold)

    dset = (dset.decode(video_decoder_cls(**decoder_kwargs), handler=wds.warn_and_continue, **additional_decoder_kwargs).
            map(reassemble, handler=wds.warn_and_continue).
            select(language_filter).
            select(unsafe_filter).
            select(aesthetics_filter).
            map(VideoResizer(size=resize_size, crop_size=crop_size,random_crop=random_crop,
                             key=video_key,width_key=original_width_key, height_key=original_height_key))
            .batched(batch_size,partial=drop_last,
                     collation_fn=dict_collation_fn)
            )

    return dset




def get_wds_dataset(args, preprocess_vid, is_train, epoch=0, tokenizer=None):
    """returns webdataset for training"""
    num_samples = args.train_num_samples
    shared_epoch = SharedEpoch(epoch=epoch)

    pipeline = [wds.SimpleShardList(args.train_data)]
    is_train = True

    pipeline.extend(
        [
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_video),
            wds.decode(wds.torch_video, handler=log_and_continue),
            wds.rename(video="mp4", text="txt"),
            wds.map_dict(video=preprocess_vid, text=lambda text: tokenizer(text)[0]),
            wds.to_tuple("video", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=True,
    )

    round_fn = math.floor
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_video_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, _ = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_wds_dataset(args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    return data
