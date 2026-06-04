"""CLIP vision encoder exported via the standalone inference-driven pipeline."""
import torch
import uuid
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from optimum.exporters.onnx.convert import export_models
from optimum.utils.save_utils import maybe_save_preprocessors
from onnxruntime import InferenceSession

from inference_driven_model_compiler.optimum.exporters.onnx.utils import trace_model_shapes, generate_config_dim
from inference_driven_model_compiler.optimum.exporters.onnx.model_configs import DummyOnnxConfig

ckpt = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(ckpt)
clip_model = CLIPModel.from_pretrained(ckpt)
vision_model = clip_model.vision_model   # encoder-only sub-module
vision_model.eval()

dummy_image = torch.randint(0, 256, (224, 224, 3)).numpy()
encoded_input = processor(images=dummy_image, return_tensors="pt")
# encoded_input = {"pixel_values": tensor(1,3,224,224)}

# ── Trace shapes via the vision encoder ──────────────────────────────────
inputs, outputs = trace_model_shapes(vision_model, dict(encoded_input))
print("traced inputs :", inputs.keys())
print("traced outputs:", outputs.keys())

config_dim = generate_config_dim(clip_model, ["projection_dim", "hidden_size"])
onnx_cfg = DummyOnnxConfig(
    config=clip_model.config.vision_config,
    task="backbone",
    model_inputs=inputs,
    model_outputs=outputs,
    config_dim=config_dim,
)

# ── Export ───────────────────────────────────────────────────────────────
save_dir = Path(f"/tmp/clip_{uuid.uuid4().hex[:8]}")
save_dir.mkdir(parents=True, exist_ok=True)
export_models(
    models_and_onnx_configs={"transformer": (vision_model, onnx_cfg)},
    opset=onnx_cfg.DEFAULT_ONNX_OPSET,
    output_dir=save_dir,
    output_names=["transformer.onnx"],
)

# ── ORT inference ─────────────────────────────────────────────────────────
session = InferenceSession(str(save_dir / "transformer.onnx"))
pixel_values = encoded_input["pixel_values"].numpy()
ort_outputs = session.run(None, {"pixel_values": pixel_values})
print("last_hidden_state shape:", ort_outputs[0].shape)
