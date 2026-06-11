"""Inference-driven ONNX Runtime pipeline for image-text-to-text (VLM) models.

This is a **self-contained, additive** module — it does not modify or depend on
the existing OnTheFly/diffusion ORT classes. It targets Qwen-VL–style models
(``model.model.visual`` + ``model.model.language_model`` + ``model.lm_head``,
e.g. Qwen2-VL / Qwen2.5-VL / Qwen3.5-MoE-VL).

Design (proven equivalent to the PyTorch model, see the seam spike):

    input_ids ──embed_tokens.onnx──► inputs_embeds ┐
    pixel_values, grid ──vision_encoder.onnx──► image_embeds ─┤ scatter at
                                                              │ image-token
                                                              ▼ positions
              decoder.onnx(inputs_embeds, attention_mask, position_ids) ─► logits

``position_ids`` are the model's native 3-D mRoPE indices, computed at inference
by the ported ``get_rope_index`` (config-only, no weights).

Scope of this first milestone (intentionally minimal):
  * single image, batch size 1
  * **fixed resolution** chosen at export via ``image_size`` (the vision graph is
    shape-specialised; the Qwen-VL tower bakes ``grid_thw`` into the trace)
  * greedy decoding, no KV cache (each step re-runs the decoder over the full
    sequence — correct, just not yet optimised)

Usage::

    from optimum.onnxruntime.modeling_image_text_to_text import ORTModelForImageTextToText
    m = ORTModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", image_size=196)
    print(m.generate(image=pil_image, text="What is in this image?", max_new_tokens=32))
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _make_rope_index_fn(pt_model):
    """Bind the model's native ``get_rope_index`` to a tiny config-only shim so it
    can be called at inference without keeping the (heavy) torch model around."""
    inner = pt_model.model  # Qwen2VLModel-like
    rope_method = type(inner).get_rope_index  # unbound; only uses self.config

    class _Shim:
        def __init__(self, config):
            self.config = config

    shim = _Shim(pt_model.config)
    return rope_method.__get__(shim, _Shim)


# ── export wrappers ────────────────────────────────────────────────────────────

class _EmbedWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed = model.get_input_embeddings()

    def forward(self, input_ids):
        return self.embed(input_ids)


class _VisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.model.visual

    def forward(self, pixel_values, image_grid_thw):
        return self.visual(pixel_values, grid_thw=image_grid_thw)


class _DecoderWrapper(torch.nn.Module):
    """Full model forward over inputs_embeds (no input_ids / pixel_values), no cache.

    ``attention_mask`` is a precomputed **4-D** additive mask (B, 1, Q, KV). A 4-D
    mask makes transformers return it as-is (``_preprocess_mask_arguments`` early
    exit) instead of building one with ``torch.vmap`` — which the torchscript ONNX
    exporter cannot trace.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits


def _causal_4d(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Standard causal additive mask of shape (1, 1, seq_len, seq_len): 0 where a
    query may attend to a key, ``finfo.min`` above the diagonal."""
    min_val = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), min_val, dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m[None, None]


class ORTModelForImageTextToText:
    """ONNX Runtime image-text-to-text pipeline (Qwen-VL family, fixed resolution)."""

    def __init__(self, *, embed_session, vision_session, decoder_session, processor,
                 config, rope_index_fn, image_token_id, eos_token_ids, export_grid_thw):
        self.embed_session = embed_session
        self.vision_session = vision_session
        self.decoder_session = decoder_session
        self.processor = processor
        self.config = config
        self._rope_fn = rope_index_fn
        self.image_token_id = image_token_id
        self.eos_token_ids = set(eos_token_ids)
        self.export_grid_thw = export_grid_thw  # the grid the vision graph was traced for

    # ── export + load ──────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_id, image_size: int = 196, export_dir: str | None = None,
                        dtype: torch.dtype = torch.float32, providers=("CPUExecutionProvider",),
                        trust_remote_code: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from onnxruntime import InferenceSession

        if image_size % 28 != 0:
            raise ValueError(f"image_size must be a multiple of 28 (patch*merge); got {image_size}")

        export_dir = Path(export_dir or (Path("/dev/shm") / f"ort_itt_{Path(model_id).name}_{image_size}"))
        export_dir.mkdir(parents=True, exist_ok=True)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        # Force a fixed image resolution -> deterministic grid for the static vision graph.
        cls._force_fixed_resolution(processor, image_size)

        # eager attention: the torchscript ONNX exporter cannot convert SDPA with
        # grouped-query attention (enable_gqa=True), which Qwen-VL uses in both the
        # vision tower and the language model. This mirrors what the inference-driven
        # CLI export does (main_export forces attn_implementation="eager").
        pt = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        ).eval()
        config = pt.config
        image_token_id = config.image_token_id

        # representative input at the chosen resolution (used to trace shapes)
        ex = cls._build_inputs(processor, _gray_image(image_size),
                               "Describe the image.", image_token_id)
        export_grid_thw = ex["image_grid_thw"].tolist()

        paths = {k: str(export_dir / f"{k}.onnx") for k in ("embed_tokens", "vision_encoder", "decoder")}

        # --- export the three graphs (torchscript exporter, opset 18) ---
        with torch.no_grad():
            torch.onnx.export(
                _EmbedWrapper(pt), (ex["input_ids"],), paths["embed_tokens"],
                input_names=["input_ids"], output_names=["inputs_embeds"],
                dynamic_axes={"input_ids": {1: "seq"}, "inputs_embeds": {1: "seq"}},
                opset_version=18, dynamo=False,
            )
            torch.onnx.export(
                _VisionWrapper(pt), (ex["pixel_values"], ex["image_grid_thw"]), paths["vision_encoder"],
                input_names=["pixel_values", "image_grid_thw"], output_names=["image_embeds"],
                opset_version=18, dynamo=False,  # static: fixed resolution
            )
            embeds = pt.get_input_embeddings()(ex["input_ids"])
            rope_index_fn = _make_rope_index_fn(pt)
            pos, _ = rope_index_fn(ex["input_ids"], ex["image_grid_thw"], None, ex["attention_mask"])
            seq = ex["input_ids"].shape[1]
            mask4d = _causal_4d(seq, dtype)
            torch.onnx.export(
                _DecoderWrapper(pt), (embeds, mask4d, pos), paths["decoder"],
                input_names=["inputs_embeds", "attention_mask", "position_ids"],
                output_names=["logits"],
                dynamic_axes={"inputs_embeds": {1: "seq"}, "attention_mask": {2: "q", 3: "kv"},
                              "position_ids": {2: "seq"}, "logits": {1: "seq"}},
                opset_version=18, dynamo=False,
            )

        eos_ids = _eos_ids(config, getattr(pt, "generation_config", None))
        del pt  # heavy torch model no longer needed

        so = {"providers": list(providers)}
        sessions = {k: InferenceSession(p, **so) for k, p in paths.items()}

        return cls(
            embed_session=sessions["embed_tokens"], vision_session=sessions["vision_encoder"],
            decoder_session=sessions["decoder"], processor=processor, config=config,
            rope_index_fn=rope_index_fn, image_token_id=image_token_id,
            eos_token_ids=eos_ids, export_grid_thw=export_grid_thw,
        )

    # ── generation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image, text, max_new_tokens: int = 32, return_text: bool = True):
        inp = self._build_inputs(self.processor, image, text, self.image_token_id)
        if inp["image_grid_thw"].tolist() != self.export_grid_thw:
            raise ValueError(
                f"Image grid {inp['image_grid_thw'].tolist()} != export grid {self.export_grid_thw}. "
                "Pass an image that resizes to the export resolution (fixed-resolution pipeline)."
            )
        input_ids = inp["input_ids"]
        attention_mask = inp["attention_mask"]

        # vision once
        image_embeds = self._run(self.vision_session, {
            "pixel_values": _np(inp["pixel_values"], np.float32),
            "image_grid_thw": _np(inp["image_grid_thw"], np.int64),
        })[0]
        # embed prompt + scatter image embeds at image-token positions
        embeds = self._run(self.embed_session, {"input_ids": _np(input_ids, np.int64)})[0]
        mask = (input_ids.numpy() == self.image_token_id)
        embeds[mask] = image_embeds.astype(embeds.dtype)

        cur_ids = input_ids
        generated = []
        for _ in range(max_new_tokens):
            pos, _ = self._rope_fn(cur_ids, inp["image_grid_thw"], None, attention_mask)
            mask4d = _causal_4d(cur_ids.shape[1], torch.float32).numpy()
            logits = self._run(self.decoder_session, {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": mask4d,
                "position_ids": _np(pos, np.int64),
            })[0]
            next_id = int(logits[0, -1].argmax())
            if next_id in self.eos_token_ids:
                break
            generated.append(next_id)
            # append the new token's embedding / ids / mask
            new_embed = self._run(self.embed_session,
                                  {"input_ids": np.array([[next_id]], np.int64)})[0]
            embeds = np.concatenate([embeds, new_embed.astype(embeds.dtype)], axis=1)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1)

        if return_text:
            return self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return generated

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _run(session, feed):
        return session.run(None, feed)

    @staticmethod
    def _force_fixed_resolution(processor, image_size):
        ip = processor.image_processor
        px = image_size * image_size
        for attr in ("min_pixels", "max_pixels"):
            if hasattr(ip, attr):
                setattr(ip, attr, px)
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size = {**ip.size, "shortest_edge": px, "longest_edge": px}

    @staticmethod
    def _build_inputs(processor, image, text, image_token_id):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        out = processor(text=[chat], images=[image], return_tensors="pt")
        return dict(out)


# module-level small helpers (kept out of the class to stay torch.onnx-friendly)

def _np(t, dtype):
    return t.cpu().numpy().astype(dtype) if isinstance(t, torch.Tensor) else np.asarray(t, dtype)


def _gray_image(size):
    from PIL import Image
    return Image.new("RGB", (size, size), (120, 130, 140))


def _eos_ids(config, gen_config):
    ids = set()
    for src in (gen_config, config):
        e = getattr(src, "eos_token_id", None)
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, (list, tuple)):
            ids.update(int(x) for x in e)
    return ids or {0}
