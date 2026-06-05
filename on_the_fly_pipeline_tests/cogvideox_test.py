import torch
from diffusers.utils import export_to_video

from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import ORTDiffusionPipeline

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

inf_kwargs = {
    "prompt": "A panda eating bamboo in a lush green forest, realistic",
    "negative_prompt": "low quality, blurred, static, worst quality",
    "height": 480,
    "width": 720,
    "num_frames": 49,
    "num_inference_steps": 50,
    "guidance_scale": 6.0,
}

module_fixed_dynamic_axis = {
    "text_encoder": ["d_model", "vocab_size"],
    "transformer":  ["in_channels", "hidden_size"],
    "vae_decoder":  ["latent_channels"],
}

pipe = ORTDiffusionPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    provider=providers[0],
    torch_dtype=torch.float16,
    inference_kwargs=inf_kwargs,
    module_fixed_axis_fields=module_fixed_dynamic_axis,
    export_by_inference=True,
)

print("Loaded successfully on:", pipe.device)

output = pipe(**inf_kwargs).frames[0]
export_to_video(output, "cogvideox_output.mp4", fps=8)
print("Saved cogvideox_output.mp4")
