import torch
from diffusers.utils import export_to_video

from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import ORTDiffusionPipeline

wan_list = [
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ali-vilab/text-to-video-ms-1.7b",
]

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

inf_kwargs = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "height": 240,
    "width":416,
    "num_frames": 21,
    "guidance_scale": 5.0
}

module_fixed_dynamic_axis = {
    "text_encoder": ["d_model", "vocab_size"],
    "transformer": ["in_channels", "text_dim"],
    "vae_decoder": ["base_dim", "z_dim"],
    "vae_encoder": ["base_dim", "z_dim"],
}


pipe = ORTDiffusionPipeline.from_pretrained(
    wan_list[0],
    provider=providers[0],  # Force GPU
    torch_dtype=torch.float16,
    inference_kwargs=inf_kwargs,
    module_fixed_axis_fields=module_fixed_dynamic_axis,
    export_by_inference=True,
)

print("Loaded successfully on:", pipe.device)
prompt = "A cat walks on the grass grass grass"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(**inf_kwargs).frames[0]
export_to_video(output, "output.mp4", fps=15)