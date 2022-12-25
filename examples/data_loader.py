"""Reader module provides files and webdataset readers"""
import io
import torchvision
import tempfile

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def create_webdataset(
    urls,
    video_transform,
    enable_text=True,
    enable_video=True,
    video_key="mp4",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    input_sampler=lambda a: a,
):
    """Create a WebDataset reader, it can read a webdataset of video, text and json"""
    import clip  # pylint: disable=import-outside-toplevel
    import webdataset as wds  # pylint: disable=import-outside-toplevel

    urls = input_sampler(urls)

    dataset = wds.WebDataset(urls, cache_dir=cache_path, cache_size=10**10, handler=wds.handlers.warn_and_continue)
    tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_video and video_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False
        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        if enable_video:
            video_data = item[video_key]
            with tempfile.NamedTemporaryFile() as f:
                f.write(video_data)
                # video = torchvision.io.read_video(video_data)
                video, audio, meta = torchvision.io.read_video(f.name)
            video_tensor = video_transform(video)
            output["video_filename"] = item["__key__"]
            output["video_tensor"] = video_tensor

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)
            output["text_tokens"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data



class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        sampler,
        preprocess,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_video=True,
        enable_metadata=False,
        wds_video_key="mp4",
        wds_caption_key="txt",
        cache_path=None,
    ):
        self.batch_size = batch_size
        dataset = create_webdataset(
            input_dataset,
            preprocess,
            enable_text=enable_text,
            enable_video=enable_video,
            video_key=wds_video_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
            input_sampler=sampler,
        )
        self.dataloader = dataset_to_dataloader(dataset, batch_size, num_prepro_workers, "webdataset")

    def __iter__(self):
        for batch in self.dataloader:
            yield batch
