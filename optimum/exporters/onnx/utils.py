from __future__ import annotations

import inspect
from typing import Any

import torch


def generate_config_dim(model, dim_names: list[str] | None) -> dict[str, int]:
    if not dim_names:
        return {}
    return {k: getattr(model.config, k) for k in dim_names if hasattr(model.config, k)}


def _flatten_output(output) -> dict[str, tuple]:
    """Extract {name: shape} from a model output (dataclass, dict, or tensor)."""
    from dataclasses import fields, is_dataclass
    result = {}
    if torch.is_tensor(output):
        result["output"] = tuple(output.shape)
    elif is_dataclass(output):
        for f in fields(output):
            val = getattr(output, f.name)
            if torch.is_tensor(val):
                result[f.name] = tuple(val.shape)
    elif isinstance(output, dict):
        for k, v in output.items():
            if torch.is_tensor(v):
                result[k] = tuple(v.shape)
    elif isinstance(output, (list, tuple)):
        for i, v in enumerate(output):
            if torch.is_tensor(v):
                result[f"output_{i}"] = tuple(v.shape)
    return result


def trace_model_shapes(
    model,
    inf_kwargs: dict[str, Any],
    skip_random_generation: bool = False,
) -> tuple[dict[str, Any], dict[str, tuple]]:
    """Run inference to trace input/output tensor shapes.

    - Encoder-only models: single forward pass.
    - Encoder-decoder models: only the encoder is traced (for feature extraction).
    - Decoder-only models with KV cache: prefill + decode step, including
      position_ids and past_key_values.

    Returns:
        inputs : {name: tensor_or_shape}  (flat; KV entries as past_key_values.i.key/value)
        outputs: {name: shape_tuple}
    """
    def _store(val):
        return val if skip_random_generation else tuple(val.shape)

    fwd_params = inspect.signature(model.forward).parameters
    is_enc_dec = getattr(getattr(model, "config", None), "is_encoder_decoder", False)

    # ── Encoder-decoder: export encoder only ─────────────────────────────
    if is_enc_dec and hasattr(model, "encoder"):
        enc_sig = set(inspect.signature(model.encoder.forward).parameters.keys())
        enc_kwargs = {k: v for k, v in inf_kwargs.items() if k in enc_sig}
        with torch.no_grad():
            enc_out = model.encoder(**enc_kwargs)
        inputs  = {k: _store(v) for k, v in enc_kwargs.items() if torch.is_tensor(v)}
        outputs = _flatten_output(enc_out)
        return inputs, outputs

    # ── Standard prefill ─────────────────────────────────────────────────
    prefill_kwargs = dict(inf_kwargs)
    with torch.no_grad():
        prefill_out = model(**prefill_kwargs)

    past_key_values = getattr(prefill_out, "past_key_values", None)

    if past_key_values is not None and len(past_key_values) > 0:
        # ── Decoder-only: add decode step with KV cache ───────────────────
        ids = prefill_kwargs["input_ids"]
        batch, seq = ids.shape
        new_token = torch.zeros((batch, 1), dtype=ids.dtype)
        new_attn  = torch.ones((batch, seq + 1), dtype=torch.long)
        new_pos   = torch.full((batch, 1), seq, dtype=torch.long)

        decode_kwargs: dict[str, Any] = {
            "input_ids":       new_token,
            "attention_mask":  new_attn,
            "past_key_values": past_key_values,
        }
        if "position_ids" in fwd_params:
            decode_kwargs["position_ids"] = new_pos

        with torch.no_grad():
            decode_out = model(**decode_kwargs)

        inputs = {k: _store(v) for k, v in decode_kwargs.items() if torch.is_tensor(v)}
        for i, (k, v) in enumerate(decode_kwargs["past_key_values"]):
            inputs[f"past_key_values.{i}.key"]   = _store(k)
            inputs[f"past_key_values.{i}.value"] = _store(v)

        outputs: dict[str, tuple] = {}
        if getattr(decode_out, "logits", None) is not None:
            outputs["logits"] = tuple(decode_out.logits.shape)
        updated_pkv = getattr(decode_out, "past_key_values", None)
        if updated_pkv is not None:
            for i, (k, v) in enumerate(updated_pkv):
                outputs[f"past_key_values.{i}.key"]   = tuple(k.shape)
                outputs[f"past_key_values.{i}.value"] = tuple(v.shape)
    else:
        # ── Encoder-only: single forward ──────────────────────────────────
        inputs  = {k: _store(v) for k, v in prefill_kwargs.items() if torch.is_tensor(v)}
        outputs = _flatten_output(prefill_out)

    return inputs, outputs
