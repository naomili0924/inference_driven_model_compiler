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
  * greedy decoding **with a KV cache**: one merged decoder graph serves both the
    prefill pass (0-length past) and incremental single-token decode (past length P)

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


class _CachedDecoderWrapper(torch.nn.Module):
    """Full model forward over inputs_embeds with a KV cache.

    Takes the past key/values flattened as positional tensors and returns
    ``(logits, *present)`` flattened the same way. One graph serves both phases:
      * prefill: past length 0 (0-length tensors), query = prompt length
      * decode:  past length P,                  query = 1
    ``attention_mask`` is a precomputed **4-D** additive mask (B, 1, Q, KV=P+Q); a
    4-D mask makes transformers skip the ``torch.vmap`` mask path that the
    torchscript ONNX exporter cannot trace.
    """
    def __init__(self, model, n_layers):
        super().__init__()
        self.model = model
        self.n_layers = n_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, *past):
        from transformers import DynamicCache
        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.n_layers))
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        flat = []
        for layer in out.past_key_values.layers:
            flat.extend([layer.keys, layer.values])
        return (out.logits, *flat)


def _causal_4d_kv(q_len: int, kv_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Causal additive mask (1, 1, q_len, kv_len) for a KV cache of length P=kv-q.

    Query i (absolute position P+i) may attend to key j iff ``j <= P+i``; masked
    entries are ``finfo.min``. Prefill (P=0) → triangular; decode (q=1) → all-zero.
    """
    min_val = torch.finfo(dtype).min
    offset = kv_len - q_len
    i = torch.arange(q_len).unsqueeze(1)
    j = torch.arange(kv_len).unsqueeze(0)
    allowed = j <= (offset + i)
    m = torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), min_val, dtype=dtype))
    return m[None, None]


def _cache_dims(config):
    tc = getattr(config, "text_config", config)
    n_layers = getattr(tc, "num_hidden_layers", getattr(config, "num_hidden_layers"))
    n_heads = getattr(tc, "num_attention_heads", getattr(config, "num_attention_heads"))
    n_kv = getattr(tc, "num_key_value_heads", n_heads)
    hidden = getattr(tc, "hidden_size", getattr(config, "hidden_size"))
    head_dim = getattr(tc, "head_dim", hidden // n_heads)
    return int(n_layers), int(n_kv), int(head_dim)


class ORTModelForImageTextToText:
    """ONNX Runtime image-text-to-text pipeline (Qwen-VL family, fixed resolution)."""

    def __init__(self, *, embed_session, vision_session, decoder_session, processor,
                 config, rope_index_fn, image_token_id, eos_token_ids, export_grid_thw,
                 n_layers, n_kv_heads, head_dim, image_size):
        self.embed_session = embed_session
        self.vision_session = vision_session
        self.decoder_session = decoder_session
        self.processor = processor
        self.config = config
        self._rope_fn = rope_index_fn
        self.image_token_id = image_token_id
        self.eos_token_ids = set(eos_token_ids)
        self.image_size = image_size            # the export target the vision graph was built for
        self.export_grid_thw = export_grid_thw  # the resolved grid (from the processor) for that target
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    # ── export + load ──────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, model_id, image_size: int = 196, export_dir: str | None = None,
                        dtype: torch.dtype = torch.float32, providers=("CPUExecutionProvider",),
                        trust_remote_code: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from onnxruntime import InferenceSession

        # image_size is a *target* pixel budget, not a hard constraint: the Qwen
        # processor's smart_resize maps any value to a valid grid (snapping each
        # side to a multiple of patch*merge=28). We export the vision graph for
        # whatever grid that representative input produces — no feature resizing.

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
            # --- decoder with KV cache: trace one real decode step (q=1, past=prompt) ---
            rope_index_fn = _make_rope_index_fn(pt)
            n_layers, n_kv, head_dim = _cache_dims(config)
            prompt_embeds = pt.get_input_embeddings()(ex["input_ids"]).clone()
            ie = pt.model.visual(ex["pixel_values"], grid_thw=ex["image_grid_thw"])
            prompt_embeds[ex["input_ids"] == image_token_id] = ie.to(prompt_embeds.dtype)
            seq = ex["input_ids"].shape[1]
            pos_prefill, _ = rope_index_fn(ex["input_ids"], ex["image_grid_thw"], None, ex["attention_mask"])
            out0 = pt(inputs_embeds=prompt_embeds, attention_mask=_causal_4d_kv(seq, seq, dtype),
                      position_ids=pos_prefill, use_cache=True)
            past_flat = []
            for layer in out0.past_key_values.layers:
                past_flat.extend([layer.keys.detach(), layer.values.detach()])
            dec_embed = prompt_embeds[:, -1:, :].detach()          # (1, 1, H)
            dec_pos = (pos_prefill[:, :, -1:] + 1).detach()        # (3, 1, 1)
            dec_mask = _causal_4d_kv(1, seq + 1, dtype)            # (1, 1, 1, seq+1)

            past_names = [f"past_key_values.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
            present_names = [f"present.{i}.{kv}" for i in range(n_layers) for kv in ("key", "value")]
            dyn = {"inputs_embeds": {1: "q"}, "attention_mask": {2: "q", 3: "kv"},
                   "position_ids": {2: "q"}, "logits": {1: "q"}}
            for nm in past_names:
                dyn[nm] = {2: "past"}
            for nm in present_names:
                dyn[nm] = {2: "kv"}
            torch.onnx.export(
                _CachedDecoderWrapper(pt, n_layers),
                (dec_embed, dec_mask, dec_pos, *past_flat), paths["decoder"],
                input_names=["inputs_embeds", "attention_mask", "position_ids", *past_names],
                output_names=["logits", *present_names],
                dynamic_axes=dyn, opset_version=18, dynamo=False,
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
            n_layers=n_layers, n_kv_heads=n_kv, head_dim=head_dim, image_size=image_size,
        )

    # ── generation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, image, text, max_new_tokens: int = 32, return_text: bool = True):
        # Snap the input image to the export target so its grid matches the static
        # vision graph (this is plain image resizing, not feature resizing).
        image = image.convert("RGB").resize((self.image_size, self.image_size))
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

        # --- prefill: past length 0, query = whole prompt ---
        seq = input_ids.shape[1]
        pos, _ = self._rope_fn(input_ids, inp["image_grid_thw"], None, attention_mask)
        feed = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": _causal_4d_kv(seq, seq, torch.float32).numpy(),
            "position_ids": _np(pos, np.int64),
        }
        feed.update(self._empty_past())
        outs = self._run(self.decoder_session, feed)
        logits, present = outs[0], outs[1:]
        next_id = int(logits[0, -1].argmax())

        cur_ids = input_ids
        generated = []
        past_len = seq
        for _ in range(max_new_tokens):
            if next_id in self.eos_token_ids:
                break
            generated.append(next_id)
            cur_ids = torch.cat([cur_ids, torch.tensor([[next_id]])], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1)
            # --- decode: feed only the new token + the cached past ---
            new_embed = self._run(self.embed_session, {"input_ids": np.array([[next_id]], np.int64)})[0]
            pos, _ = self._rope_fn(cur_ids, inp["image_grid_thw"], None, attention_mask)
            feed = {
                "inputs_embeds": new_embed.astype(np.float32),
                "attention_mask": _causal_4d_kv(1, past_len + 1, torch.float32).numpy(),
                "position_ids": _np(pos[:, :, -1:], np.int64),
            }
            feed.update(self._past_feed(present))
            outs = self._run(self.decoder_session, feed)
            logits, present = outs[0], outs[1:]
            next_id = int(logits[0, -1].argmax())
            past_len += 1

        if return_text:
            return self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        return generated

    # ── KV-cache helpers ─────────────────────────────────────────────────────────

    def _past_names(self):
        return [f"past_key_values.{i}.{kv}" for i in range(self.n_layers) for kv in ("key", "value")]

    def _empty_past(self):
        z = np.zeros((1, self.n_kv_heads, 0, self.head_dim), dtype=np.float32)
        return {name: z for name in self._past_names()}

    def _past_feed(self, present):
        # present (session outputs after logits) is in the same order as the past inputs
        return {name: present[i].astype(np.float32) for i, name in enumerate(self._past_names())}

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
