"""video dataset creation"""
import webdataset as wds
from functools import partial
from typing import List, Union

from .custom_wds import WebDatasetWithChangedDecoder, dict_collation_fn
from .transform import VideoResizer, CutsAdder, CustomTransforms
from .video_decode import VideoDecorder, VideoDecorderWithCutDetection
from .filters import (
    KeyFilter,
    LanguageFilter,
    AestheticsFilter,
    UnsafeFilter,
    UnusedKeyFilter,
)  # pylint: disable=unused-import


def reassemble(x):
    """
    Process a dictionary by updating its values based on certain conditions.

    :param dict x: The input dictionary to process.
    :return: The processed dictionary.
    :rtype: dict
    """
    new_dict = {}

    for key in x:
        if key not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            continue

        # this is updating the output of video decoders
        if isinstance(x[key], tuple) and len(x[key]) == 2:
            new_dict.update({f"{subk}": x[key][-1][subk] for subk in x[key][-1]})

        x[key] = x[key][0]
    x.update(new_dict)
    del new_dict
    return x


def get_video_dataset(
    urls,
    batch_size,
    shuffle=0,
    repeat=1,
    drop_last=False,
    video_key="mp4",
    cuts_key=None,
    decoder_kwargs=None,
    custom_transforms=None,
    aesthetics_threshold=None,
    allowed_languages=None,
    p_unsafe_threshold=None,
    resize_size=None,
    crop_size=None,
    random_crop=False,
    original_height_key="original_height",
    original_width_key="original_width",
    keys_to_remove: Union[int, List[int], None] = None,
    enforce_additional_keys=None,
):

    """
    Generates a webdataset given the specified parameters.

    Parameters:
        urls (str): The path to the dataset.
        batch_size (int): The number of samples per batch.
        shuffle (int, optional): Shuffle buffer size. Default is 0 means no shuffling.
        repeat (int, optional): Whether to repeat the dataset. Default is 1. -1 means repeating infinitely
        drop_last (bool, optional): Whether to drop the last incomplete batch. Default is False.
        video_key (str, optional): The key for video files. Default is 'mp4'.
        cuts_key (str, optional): The key for cut detection. Default is None.
        decoder_kwargs (dict, optional): Keyword arguments for the video decoder. Default is an empty dictionary.
        custom_transforms (dict, optional): Pairs of additional custom transforms to apply to samples.
        aesthetics_threshold (float, optional): Aesthetic threshold for filtering. Default is None.
        allowed_languages (list, optional): List of allowed languages. Default is None.
        p_unsafe_threshold (float, optional): Probability threshold for unsafe content filtering. Default is None.
        resize_size (tuple, optional): Tuple of (width, height) for resizing the video. Default is None.
        crop_size (tuple, optional): Tuple of (width, height) for cropping the video. Default is None.
        random_crop (bool, optional): Whether to apply random cropping. Default is False.
        original_height_key (str, optional): The key for the original video height. Default is 'original_height'.
        original_width_key (str, optional): The key for the original video width. Default is 'original_width'.
        enforce_additional_keys (list, optional): Which keys must be in each sample
        keys_to_remove ((list, int), optional): Keys which, for the sake of speed, will be
            removed before decoding. Default is None which means nothing will be removed.
    Returns:
        WebDataset: The processed webdataset.
    """

    if decoder_kwargs is None:
        decoder_kwargs = {}
    if enforce_additional_keys is None:
        enforce_additional_keys = ["txt"]
    if keys_to_remove is None:
        keys_to_remove = []

    additional_decoder_kwargs = {}
    if cuts_key:
        dataset_cls = WebDatasetWithChangedDecoder
        video_decoder_cls = partial(VideoDecorderWithCutDetection, cuts_key=cuts_key)
        additional_decoder_kwargs = {"passthrough_keys": [video_key]}
    elif decoder_kwargs == {}:  # nothing means just read the bytes
        dataset_cls = wds.WebDataset
        video_decoder_cls = None
    else:
        dataset_cls = wds.WebDataset
        video_decoder_cls = VideoDecorder  # type: ignore

    dset = dataset_cls(urls, nodesplitter=wds.split_by_node, shardshuffle=shuffle, handler=wds.warn_and_continue)

    dset = dset.repeat(repeat).shuffle(shuffle, initial=shuffle)

    unused_key_filter = UnusedKeyFilter(keys=keys_to_remove)
    dset = dset.map(unused_key_filter, handler=wds.warn_and_continue)

    # TODO: organize this such that you don't always need video.
    # should work with audio-text, just text or whatever you might want
    enforce_keys = [video_key] + enforce_additional_keys
    key_filter = KeyFilter(enforce_keys)
    dset = dset.select(key_filter)

    if cuts_key:
        cut_adder = CutsAdder(cuts_key=cuts_key, video_key=video_key)
        dset = dset.map(cut_adder, handler=wds.warn_and_continue)

    aesthetics_filter = AestheticsFilter(aesthetic_thld=aesthetics_threshold)
    language_filter = LanguageFilter(languages=allowed_languages)
    unsafe_filter = UnsafeFilter(p_unsafe_threshold=p_unsafe_threshold)
    # TODO: in the futuer only include filters we want to use based on params
    filters = [aesthetics_filter, language_filter, unsafe_filter]

    # Decoding
    if video_decoder_cls is not None:
        dset = dset.decode(
            video_decoder_cls(**decoder_kwargs), handler=wds.warn_and_continue, **additional_decoder_kwargs
        ).map(reassemble, handler=wds.warn_and_continue)

    # Filters
    for fltr in filters:
        dset = dset.select(fltr)

    # Resizing
    if decoder_kwargs != {}:  # bytes
        dset = dset.map(
            VideoResizer(
                size=resize_size,
                crop_size=crop_size,
                random_crop=random_crop,
                key=video_key,
                width_key=original_width_key,
                height_key=original_height_key,
            ),
            handler=wds.warn_and_continue,
        )

    if custom_transforms:
        dset = dset.map(CustomTransforms(custom_transforms), handler=wds.warn_and_continue)

    if decoder_kwargs != {}:
        dset = dset.batched(batch_size, partial=drop_last, collation_fn=dict_collation_fn)

    return dset
