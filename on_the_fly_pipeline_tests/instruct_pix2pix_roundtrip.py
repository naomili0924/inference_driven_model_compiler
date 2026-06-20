"""Round-trip test: persist the exported ONNX pipeline and reload it without re-exporting.

Assumes the on-the-fly export already ran (files in /dev/shm). Steps:
  1. Load from the raw export dir with export=False (no re-export).
  2. save_pretrained() to a real on-disk directory.
  3. Reload from that directory with export=False.
  4. Run inference and save the edited image.
"""

import io
import time

import torch

from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTImageEditPipeline

EXPORT_DIR = "/dev/shm"                       # where export_by_inference wrote the ONNX
SAVE_DIR = "/workspace/ip2p-onnx"             # persistent on-disk copy
provider = "CUDAExecutionProvider"


def _load_input_image():
    from PIL import Image
    url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
    try:
        import requests
        raw = requests.get(url, stream=True, timeout=30).content
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        print(f"Could not download example image ({exc}); using a synthetic one.")
        import numpy as np
        grad = np.linspace(0, 255, 512, dtype=np.uint8)
        arr = np.stack([np.tile(grad, (512, 1)), np.tile(grad[::-1], (512, 1)),
                        np.tile(grad.reshape(-1, 1), (1, 512))], axis=-1)
        img = Image.fromarray(arr, "RGB")
    return img.resize((512, 512))


image = _load_input_image()
inf_kwargs = {
    "prompt": "turn him into a cyborg",
    "image": image,
    "num_inference_steps": 10,
    "image_guidance_scale": 1.5,
    "guidance_scale": 7.5,
}

# 1. Load the already-exported graphs (no re-export).
t0 = time.time()
pipe = OnTheFlyORTImageEditPipeline.from_pretrained(
    EXPORT_DIR, export=False, provider=provider, torch_dtype=torch.float32,
)
print(f"[1] Loaded from {EXPORT_DIR} (export=False) in {time.time()-t0:.1f}s "
      f"as {type(pipe).__name__}")

# 2. Persist to real disk.
t0 = time.time()
pipe.save_pretrained(SAVE_DIR)
print(f"[2] save_pretrained -> {SAVE_DIR} in {time.time()-t0:.1f}s")

# 3. Reload from the persistent copy, again without re-exporting.
del pipe
t0 = time.time()
pipe2 = OnTheFlyORTImageEditPipeline.from_pretrained(
    SAVE_DIR, export=False, provider=provider, torch_dtype=torch.float32,
)
print(f"[3] Reloaded from {SAVE_DIR} (export=False) in {time.time()-t0:.1f}s")

# 4. Inference from the persisted copy.
edited = pipe2(**inf_kwargs).images[0]
out = "instruct_pix2pix_roundtrip_output.png"
edited.save(out)
print(f"[4] Saved {out} {edited.size}")
