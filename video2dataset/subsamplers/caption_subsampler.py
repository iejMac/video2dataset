"""
video captioner
"""
import os
import glob
import json
import argparse
import string
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BatchEncoding,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Blip2QFormerModel,
    Blip2VisionModel,
    AutoTokenizer,
)

from transformers.modeling_outputs import BaseModelOutputWithPooling

from .subsampler import Subsampler


class AttrDict(dict):
    """
    Lets us access dict keys with <dict>.key
    """

    # pylint: disable=super-with-arguments
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def process(
    processor: Blip2Processor,
    video: torch.Tensor = None,
    text = None,
) -> BatchEncoding:
    """Process videos and texts for VideoBLIP.

    :param images: a tensor of shape (batch, channel, time, height, width) or
        (channel, time, height, width)
    """
    if video is not None:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        batch, channel, time, _, _ = video.size()
        video = video.permute(0, 2, 1, 3, 4).flatten(end_dim=1)
    inputs = processor(images=video, text=text, return_tensors="pt")
    if video is not None:
        _, _, height, weight = inputs.pixel_values.size()
        inputs["pixel_values"] = inputs.pixel_values.view(
            batch, time, channel, height, weight
        ).permute(0, 2, 1, 3, 4)
    return inputs


class VideoBlipVisionModel(Blip2VisionModel):
    """A simple, augmented version of Blip2VisionModel to handle videos."""

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        """Flatten `pixel_values` along the batch and time dimension, pass it
        through the original vision model, then unflatten it back.

        :param pixel_values: a tensor of shape (batch, channel, time, height, width)

        :returns:
            last_hidden_state: a tensor of shape (batch, time * seq_len, hidden_size)
            pooler_output: a tensor of shape (batch, time, hidden_size)
            hidden_states:
                a tuple of tensors of shape (batch, time * seq_len, hidden_size),
                one for the output of the embeddings + one for each layer
            attentions:
                a tuple of tensors of shape (batch, time, num_heads, seq_len, seq_len),
                one for each layer
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch, _, time, _, _ = pixel_values.size()

        # flatten along the batch and time dimension to create a tensor of shape
        # (batch * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # now restore the original dimensions
        # vision_outputs.last_hidden_state is of shape
        # (batch * time, seq_len, hidden_size)
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(
            batch, time * seq_len, -1
        )
        # vision_outputs.pooler_output is of shape
        # (batch * time, hidden_size)
        pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
        # hidden_states is a tuple of tensors of shape
        # (batch * time, seq_len, hidden_size)
        hidden_states = (
            tuple(
                hidden.view(batch, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )
        # attentions is a tuple of tensors of shape
        # (batch * time, num_heads, seq_len, seq_len)
        attentions = (
            tuple(
                hidden.view(batch, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions
            )
            if vision_outputs.attentions is not None
            else None
        )
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return (last_hidden_state, pooler_output, hidden_states, attentions)


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.vision_model = VideoBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()


class VideoBlip:
    def __init__(self, args):
        assert args.video_captioner in ["kpyu/video-blip-flan-t5-xl-ego4d", "kpyu/video-blip-opt-2.7b-ego4d"]
        model = args.video_captioner
        if args.prompt is None:
            self.prompt = "Question: Can you give a detailed description of what is happening in this video? Answer:"
        else:
            self.prompt = args.prompt

        self.device = args.get("device", "cuda")
        self.processor = Blip2Processor.from_pretrained(model)
        self.model = VideoBlipForConditionalGeneration.from_pretrained(model).to(self.device)

    def __call__(self, frames):
        video = torch.from_numpy(frames)
        video = rearrange(video, 't h w c -> c t h w')
        video = video.round().byte()
        # ([c, t, h, w]) --> 0, 255 & uint8
        inputs = process(self.processor, video=video, text=self.prompt).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, 
                max_new_tokens=100,
                num_beams=10, 
                early_stopping=True,
                do_sample=True,
                temperature=1.,
                top_k=50, 
                top_p=1.0, 
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


class CaptionSubsampler(Subsampler):
    def __init__(
        self,
        captioner_args=None,
        is_slurm_task=False,
    ):
        if is_slurm_task:
            local_rank = os.environ["LOCAL_RANK"]
            device = f"cuda:{local_rank}"
            captioner_args["device"] = device
        if not isinstance(captioner_args, AttrDict):
            captioner_args = AttrDict(captioner_args)
        self.captioner = VideoBlip(captioner_args)

    def __call__(self, frames):
        try:
            caption = self.captioner(frames)
            return caption, None
        except Exception as err:  # pylint: disable=broad-except
            return [], str(err)
