import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-prompt-processor")
class StableDiffusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)
        
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir, LoRA_path=None, ckpt_path=None):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if ckpt_path is not None:
            from diffusers import StableDiffusionPipeline
            
            pipe = StableDiffusionPipeline.from_ckpt(
                ckpt_path,
            )
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder

        else:

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
            )

            class Text_pipe:
                def __init__(self, text_encoder, tokenizer):
                    super().__init__()
                    self.text_encoder = text_encoder
                    self.tokenizer = tokenizer
                    self.device = text_encoder.device
                    self.weights_dtype = torch.float32

            tmp = Text_pipe(text_encoder, tokenizer)
            
            threestudio.info(f"prompt lora test: " + str(LoRA_path))
            if LoRA_path is not None:
                try:
                    from threestudio.utils.lora import tune_lora_scale, patch_pipe
                    patch_pipe(
                        tmp,
                        LoRA_path,
                        patch_text=True,
                        patch_ti=True,
                        patch_unet=False,
                    )
                    #threestudio.info(f"prompt lora test again: " + str(self.cfg.LoRA_path))
                    #print("load lora for prompt in:.{}".format(self.cfg.LoRA_path))
                    tune_lora_scale(tmp.text_encoder, 1.00)
                    text_encoder = tmp.text_encoder
                    tokenizer = tmp.tokenizer
                except:
                    from threestudio.utils.lora import load_lora_weights
                    tmp = load_lora_weights(
                        tmp, LoRA_path, tmp.device, tmp.weights_dtype, 
                        multiplier=1.00, patch_unet=False, patch_text=True
                    )
                    text_encoder = tmp.text_encoder
                    tokenizer = tmp.tokenizer
        
        with torch.no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    (
                        f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt" 
                        if LoRA_path is None else
                        f"{hash_prompt(pretrained_model_name_or_path, prompt)}_lora.pt"
                    ),
                ),
            )

        del text_encoder