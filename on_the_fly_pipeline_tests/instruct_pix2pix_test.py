"""Inference-driven ONNX export + image editing for InstructPix2Pix.

Mirrors ``sdxl_text_to_image_test.py`` (diffusion text-to-image): export a model
on the fly by tracing one real inference pass, then run generation entirely
through ONNX Runtime — but for an *image-editing* model.

InstructPix2Pix (``timbrooks/instruct-pix2pix``) is UNet-based like SD-1.5, with
two image-editing specifics that ``ORTImageEditPipeline`` exercises during the
single tracing pass:

  * the input image is encoded to latents through the **VAE encoder**
    (``vae.encode(image).latent_dist`` — unused by text-to-image), and
  * the **UNet** takes 8 input channels: the noisy latents concatenated with the
    encoded image latents.

Run (needs a GPU + ~4 GB download for the model):

    cd /workspace
    PYTHONPATH=/workspace python \
        inference_driven_model_compiler/on_the_fly_pipeline_tests/instruct_pix2pix_test.py
"""

import io

import torch

from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
    ORTImageEditPipeline,
)

model_id = "timbrooks/instruct-pix2pix"

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _load_input_image():
    """The canonical InstructPix2Pix example image, with a synthetic fallback."""
    from PIL import Image

    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    try:
        import requests

        raw = requests.get(url, stream=True, timeout=30).content
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # offline / unreachable — fall back to a gradient
        print(f"Could not download example image ({exc}); using a synthetic one.")
        import numpy as np

        grad = np.linspace(0, 255, 512, dtype=np.uint8)
        arr = np.stack([np.tile(grad, (512, 1)),
                        np.tile(grad[::-1], (512, 1)),
                        np.tile(grad.reshape(-1, 1), (1, 512))], axis=-1)
        img = Image.fromarray(arr, "RGB")
    # Keep dims a multiple of 8 so the VAE latent grid is clean.
    return img.resize((512, 512))


image = _load_input_image()

inf_kwargs = {
    "prompt": "turn him into a cyborg",
    "image": image,
    "num_inference_steps": 10,
    "image_guidance_scale": 1.5,
    "guidance_scale": 7.5,
}

# Treat the obvious architecture constants as fixed (static) tensor dims; the
# tracer infers the rest (batch, etc.) empirically.
module_fixed_axis_fields = {
    "text_encoder": ["hidden_size", "vocab_size"],
    "unet":         ["in_channels", "cross_attention_dim"],
    "vae_encoder":  ["latent_channels"],
    "vae_decoder":  ["latent_channels"],
}

pipe = ORTImageEditPipeline.from_pretrained(
    model_id,
    provider=providers[0],  # Force GPU
    torch_dtype=torch.float32,
    inference_kwargs=inf_kwargs,
    module_fixed_axis_fields=module_fixed_axis_fields,
    export_by_inference=True,
)

print("Loaded successfully on:", pipe.device)
print("Pipeline class:", type(pipe).__name__)

edited = pipe(**inf_kwargs).images[0]
edited.save("instruct_pix2pix_output.png")
print("Saved instruct_pix2pix_output.png", edited.size)
