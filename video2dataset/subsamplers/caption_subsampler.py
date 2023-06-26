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
    BitsAndBytesConfig
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
    video: torch.Tensor,
    text,
) -> BatchEncoding:
    """Process videos and texts for VideoBLIP.

    :param images: a tensor of shape (batch, channel, time, height, width) or
        (channel, time, height, width)
    """
    # if video is not None:
    #     if video.dim() == 4:
    #         video = video.unsqueeze(0)S
    #     batch, channel, time, _, _ = video.size()
    #     video = video.permute(0, 2, 1, 3, 4).flatten(end_dim=1)
    inputs = processor(images=None, text=text, return_tensors="pt")
    inputs['pixel_values'] = video
    # if video is not None:
    #     _, _, height, weight = inputs.pixel_values.size()
    #     inputs["pixel_values"] = inputs.pixel_values.view(
    #         batch, time, channel, height, weight
    #     ).permute(0, 2, 1, 3, 4)
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

        if ':' in self.device:
            device_map = {'': int(self.device.split(':')[-1])}
        else:
            device_map = {'': 0}
        self.model = VideoBlipForConditionalGeneration.from_pretrained(model, load_in_8bit=True, device_map=device_map)

    def __call__(self, video):
        video = video * 0.00392156862745098
        video = (video - torch.Tensor([0.48145466, 0.4578275, 0.40821073])) / torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        video = rearrange(video, 'b t h w c -> b c t h w').to(torch.float16)

        inputs = process(self.processor, video=video, text=[self.prompt] * video.shape[0]).to(self.device)
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
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        for idx in range(len(generated_text)):
            generated_text[idx] = generated_text[idx].strip()
        return generated_text


class VideoBlipAndTxt:
    def __init__(self, args):
        assert args.video_captioner in ["kpyu/video-blip-flan-t5-xl-ego4d", "kpyu/video-blip-opt-2.7b-ego4d"]
        model = args.video_captioner
        if args.prompt is None:
            self.prompt = "Question: Can you give a detailed description of what is happening in this video? Answer:"
        else:
            self.prompt = args.prompt

        self.device = args.get("device", "cuda")
        self.processor = Blip2Processor.from_pretrained(model)
        if ':' in self.device:
            device_map = {'': int(self.device.split(':')[-1])}
        else:
            device_map = {'': 0}

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        # self.model = VideoBlipForConditionalGeneration.from_pretrained(model, quantization_config=bnb_config, device_map=device_map)
        self.model = VideoBlipForConditionalGeneration.from_pretrained(model, load_in_8bit=True, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side='left')
        # self.llm = AutoModelForCausalLM.from_pretrained(args.llm, quantization_config=bnb_config, device_map=device_map)
        self.llm = AutoModelForCausalLM.from_pretrained(args.llm, load_in_8bit=True, device_map=device_map)

    def __call__(self, video, original_caption):
        video = video * 0.00392156862745098
        video = (video - torch.Tensor([0.48145466, 0.4578275, 0.40821073])) / torch.Tensor([0.26862954, 0.26130258, 0.27577711])
        video = rearrange(video, 'b t h w c -> b c t h w').to(torch.float16)

        inputs = process(self.processor, video=video, text=[self.prompt] * video.shape[0]).to(self.device)
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
        
        vblip_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        for idx in range(len(vblip_caption)):
            vblip_caption[idx] = vblip_caption[idx].strip()

        # combine vblip with txt
        all_prompts = []
        for batch_idx in range(len(original_caption)):
            curr_orig_caption = original_caption[batch_idx]
            curr_vblip_caption = vblip_caption[batch_idx]
            prompt = f"""Here are two approximately correct video captions. First one is image-only caption of middle frame and second one is a brief video caption. Return a correct combined video caption by using both of these.

            Image-only caption: a man in white shirt holding scissors near a bag of clothes .
            Brief video caption: a man is cutting a piece of cloth .
            Combined video caption: A man in white shirt is cutting a piece of cloth from a bag of clothes.

            Image-only caption: a close up view of a geforce rtx gpu on a computer .
            Brief video caption: the geforce rtx is being installed .
            Combined video caption: A close up view of a geforce rtx gpu being installed on a computer. 

            Image-only caption: an older man with a beard is sitting in front of a bush .
            Brief video caption: a bald man is standing in front of bushes .
            Combined video caption: An older bald man with a beard is sitting in front of a bush.

            Image-only caption: a man with black hair and a black shirt is looking at the camera .
            Brief video caption: a cartoon character is sitting in an office .
            Combined video caption: A cartoon of a man with black hair and a black shirt sitting in an office. He is looking at the camera.

            Image-only caption: {curr_orig_caption}
            Brief video caption: {curr_vblip_caption} .
            Combined video caption: """
            all_prompts.append(prompt)

        inputs = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        output = self.llm.generate(inputs["input_ids"], num_beams=5, max_new_tokens=100)
        combined_caption = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        for idx in range(len(combined_caption)):    
            combined_caption[idx] = combined_caption[idx].split("Combined video caption: ")[-1]
            combined_caption[idx] = combined_caption[idx].strip().lstrip('"').rstrip('"')
        return combined_caption, vblip_caption


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
        else:
            captioner_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(captioner_args, AttrDict):
            captioner_args = AttrDict(captioner_args)
        
        ### for vblip + llm
        # self.captioner = VideoBlipAndTxt(captioner_args)
        ### for vblip only
        self.captioner = VideoBlip(captioner_args)

    def __call__(self, frames, original_caption):
        try:
            ### for vblip + llm
            # combined_caption, vblip_caption = self.captioner(frames, original_caption)
            # return [combined_caption, vblip_caption], None
            ### for vblip only
            combined_caption = self.captioner(frames)
            return [combined_caption, ], None
        except Exception as err:  # pylint: disable=broad-except
            return [], str(err)
