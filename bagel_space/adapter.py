"""Load the official BAGEL checkpoint on one A100 80 GB GPU."""
from pathlib import Path
import random

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer

MODEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"
MODEL_REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
UPSTREAM_REVISION = "a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f"


class BagelInferencer(InterleaveInferencer):
    def decode_image(self, latent, image_shape):
        """Move latents to the VAE device without changing sampler precision."""
        latent = latent.to(device=next(self.vae_model.parameters()).device)
        return super().decode_image(latent, image_shape)


class BagelAdapter:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.bfloat16
        path = Path(snapshot_download(
            MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        ))
        llm = Qwen2Config.from_json_file(str(path / "llm_config.json"))
        llm.qk_norm = True
        llm.tie_word_embeddings = False
        llm.layer_module = "Qwen2MoTDecoderLayer"
        vit = SiglipVisionConfig.from_json_file(str(path / "vit_config.json"))
        vit.rope = False
        vit.num_hidden_layers -= 1
        self.vae, vae_config = load_ae(local_path=str(path / "ae.safetensors"))
        self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        config = BagelConfig(
            visual_gen=True, visual_und=True, llm_config=llm, vit_config=vit,
            vae_config=vae_config, vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh", latent_patch_size=2, max_latent_size=64,
        )
        with init_empty_weights():
            model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
        self.model = load_checkpoint_and_dispatch(
            model, checkpoint=str(path / "ema.safetensors"),
            device_map={"": "cuda:0"}, dtype=self.dtype, force_hooks=True,
        ).eval().requires_grad_(False)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(path))
        self.tokenizer, self.token_ids, _ = add_special_tokens(self.tokenizer)
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)
        self.inferencer = BagelInferencer(
            self.model, self.vae, self.tokenizer, self.vae_transform,
            self.vit_transform, self.token_ids,
        )
        self.layers = self.model.language_model.model.layers

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def metadata(self):
        return dict(
            model=MODEL_ID, model_revision=MODEL_REVISION,
            upstream_revision=UPSTREAM_REVISION, dtype=str(self.dtype),
            capture="decoder block output before final norm; packed [tokens, hidden]",
            use_cache=True,
        )

    def prepare_understanding(self, image_path):
        with Image.open(image_path) as source:
            image = pil_img2rgb(source)
        return self.vae_transform.resize_transform(image)

    @torch.no_grad()
    def forward_understanding(self, image, question):
        with self.autocast():
            context = self.inferencer.init_gen_context()
            context = self.inferencer.update_context_image(image, context, vae=False, vit=True)
            return self.inferencer.update_context_text(question, context)

    @torch.no_grad()
    def answer(self, context, max_new_tokens=128):
        with self.autocast():
            return self.inferencer.gen_text(context, max_length=max_new_tokens, do_sample=False)

    @torch.no_grad()
    def generate(self, prompt, steps=50, guidance=1.0, seed=42):
        self.seed(seed)
        return self.inferencer(
            text=prompt, think=False, image_shapes=(1024, 1024),
            num_timesteps=steps, cfg_text_scale=guidance, cfg_img_scale=1.0,
            cfg_interval=(0.4, 1.0), cfg_renorm_min=0.0,
            cfg_renorm_type="global", timestep_shift=3.0, enable_taylorseer=False,
        )["image"]
