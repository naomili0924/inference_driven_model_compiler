"""Inference-driven ONNX export + text-to-image generation for SDXL-Turbo.

Mirrors ``text_to_video_test.py`` (diffusion) and ``gpt2_text_generation.py``
(decoder-only): export a model on the fly by tracing one real inference pass,
then run generation entirely through ONNX Runtime.

SDXL is UNet-based with two CLIP text encoders and pooled micro-conditioning
(``added_cond_kwargs`` = ``text_embeds`` + ``time_ids``), exercising the
text-to-image export path.

Run (needs a GPU + ~7 GB download for the model):

    cd /workspace
    PYTHONPATH=/workspace python \
        inference_driven_model_compiler/on_the_fly_pipeline_tests/sdxl_text_to_image_test.py
"""

import torch

from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
    OnTheFlyORTDiffusionPipeline,
)

model_id = "stabilityai/sdxl-turbo"

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

prompt = "A cinematic photo of a red panda astronaut on the moon, highly detailed"

# SDXL-Turbo is a one-step distilled model: a single step with no CFG.
inf_kwargs = {
    "prompt": prompt,
    "num_inference_steps": 1,
    "guidance_scale": 0.0,
}

# Treat the obvious architecture constants as fixed (static) tensor dims; the
# tracer infers the rest (batch, etc.) empirically.
module_fixed_axis_fields = {
    "text_encoder":   ["hidden_size", "vocab_size"],
    "text_encoder_2": ["hidden_size", "vocab_size", "projection_dim"],
    "unet":           ["in_channels", "cross_attention_dim"],
    "vae_decoder":    ["latent_channels"],
}

pipe = OnTheFlyORTDiffusionPipeline.from_pretrained(
    model_id,
    provider=providers[0],  # Force GPU
    torch_dtype=torch.float16,
    inference_kwargs=inf_kwargs,
    module_fixed_axis_fields=module_fixed_axis_fields,
    export_by_inference=True,
)

print("Loaded successfully on:", pipe.device)
print("Pipeline class:", type(pipe).__name__)

image = pipe(**inf_kwargs).images[0]
image.save("sdxl_turbo_output.png")
print("Saved sdxl_turbo_output.png", image.size)
