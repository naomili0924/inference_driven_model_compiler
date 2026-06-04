from __future__ import annotations

import inspect
from typing import Any

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def generate_config_dim(model, dim_names: list[str] | None) -> dict[str, int]:
    if not dim_names:
        return {}
    return {k: getattr(model.config, k) for k in dim_names if hasattr(model.config, k)}


def _flatten_output(output) -> dict[str, tuple]:
    """Collect {name: shape} for every tensor field of a model output."""
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


def _shape_of(val: Any) -> tuple:
    return tuple(val.shape) if isinstance(val, torch.Tensor) else val


# ── input variation ───────────────────────────────────────────────────────────

def _generate_variation(inf_kwargs: dict, trial_idx: int, model) -> dict:
    """Return a modified copy of *inf_kwargs* with different tensor shapes.

    Odd trials double the batch size; even trials ≥ 2 extend text sequence
    lengths; some trials combine both.  Non-tensor entries are kept as-is.
    """
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 30000)

    # What to change for each trial index
    batch_mult   = 2 if trial_idx % 2 == 1 else 1
    seq_ext      = 7 * ((trial_idx + 1) // 2) if trial_idx >= 2 else 0

    varied: dict = {}
    for key, val in inf_kwargs.items():
        if not torch.is_tensor(val):
            varied[key] = val
            continue

        shape = list(val.shape)
        dtype = val.dtype

        # Vary batch dimension
        shape[0] = shape[0] * batch_mult

        # Vary sequence dimension for text-like tensors
        if len(shape) > 1 and seq_ext > 0:
            if any(tag in key for tag in ("input_id", "token_type", "attention_mask", "position")):
                shape[1] += seq_ext

        # Generate a tensor of the new shape with plausible values
        if dtype in (torch.long, torch.int, torch.int32, torch.int64):
            if "input_id" in key and "token_type" not in key:
                varied[key] = torch.randint(1, vocab_size, shape, dtype=dtype)
            elif "token_type" in key:
                varied[key] = torch.zeros(shape, dtype=dtype)
            elif "attention_mask" in key:
                varied[key] = torch.ones(shape, dtype=dtype)
            elif "position" in key:
                batch, seq = shape[0], shape[-1]
                varied[key] = (
                    torch.arange(seq, dtype=dtype).unsqueeze(0).expand(batch, -1)
                )
            else:
                varied[key] = torch.zeros(shape, dtype=dtype)
        else:
            varied[key] = torch.randn(shape, dtype=dtype)

    return varied


# ── single-trace helpers ──────────────────────────────────────────────────────

def _trace_encoder_decoder(model, inf_kwargs: dict) -> tuple[dict, dict] | None:
    """Run only the encoder submodule (for T5 / BART / Whisper etc.)."""
    enc_sig = set(inspect.signature(model.encoder.forward).parameters.keys())
    enc_kwargs = {k: v for k, v in inf_kwargs.items() if k in enc_sig}
    if not enc_kwargs:
        return None
    with torch.no_grad():
        enc_out = model.encoder(**enc_kwargs)
    inputs  = {k: tuple(v.shape) for k, v in enc_kwargs.items() if torch.is_tensor(v)}
    outputs = _flatten_output(enc_out)
    return inputs, outputs


def _trace_decoder_only(
    model, inf_kwargs: dict, fwd_params: dict, n_new_tokens: int = 1,
) -> tuple[dict, dict] | None:
    """Prefill + one decode step for decoder-only (GPT-style) models.

    *n_new_tokens* controls how many query tokens the decode step feeds.
    Varying it across trials lets the tracer detect that the query
    sequence-length dimension (prefill = many tokens, decode = few) is dynamic.
    """
    with torch.no_grad():
        prefill_out = model(**inf_kwargs)

    past_key_values = getattr(prefill_out, "past_key_values", None)
    if not past_key_values:
        return None                         # no KV cache → use encoder path

    ids = inf_kwargs["input_ids"]
    batch, seq = ids.shape
    new_token = torch.zeros((batch, n_new_tokens), dtype=ids.dtype)
    new_attn  = torch.ones((batch, seq + n_new_tokens), dtype=torch.long)
    new_pos   = (
        torch.arange(seq, seq + n_new_tokens, dtype=torch.long)
        .unsqueeze(0).expand(batch, -1)
    )

    decode_kwargs: dict = {
        "input_ids":       new_token,
        "attention_mask":  new_attn,
        "past_key_values": past_key_values,
    }
    if "position_ids" in fwd_params:
        decode_kwargs["position_ids"] = new_pos

    with torch.no_grad():
        decode_out = model(**decode_kwargs)

    inputs = {k: tuple(v.shape) for k, v in decode_kwargs.items() if torch.is_tensor(v)}
    for i, (k, v) in enumerate(decode_kwargs["past_key_values"]):
        inputs[f"past_key_values.{i}.key"]   = tuple(k.shape)
        inputs[f"past_key_values.{i}.value"] = tuple(v.shape)

    outputs: dict = {}
    if getattr(decode_out, "logits", None) is not None:
        outputs["logits"] = tuple(decode_out.logits.shape)
    updated_pkv = getattr(decode_out, "past_key_values", None)
    if updated_pkv:
        # Use "present.*" for output KV (standard optimum convention).
        # Sharing names with the "past_key_values.*" inputs makes torch.onnx
        # append a ".1" suffix to the inputs to disambiguate, which then breaks
        # KV pairing at generation time.
        for i, (k, v) in enumerate(updated_pkv):
            outputs[f"present.{i}.key"]   = tuple(k.shape)
            outputs[f"present.{i}.value"] = tuple(v.shape)

    return inputs, outputs


def _trace_encoder_only(model, inf_kwargs: dict) -> tuple[dict, dict]:
    """Single forward pass for encoder-only models (BERT, ViT, etc.)."""
    with torch.no_grad():
        out = model(**inf_kwargs)
    inputs  = {k: tuple(v.shape) for k, v in inf_kwargs.items() if torch.is_tensor(v)}
    outputs = _flatten_output(out)
    return inputs, outputs


def _trace_once(
    model,
    inf_kwargs: dict,
    fwd_params: dict,
    is_enc_dec: bool,
    n_new_tokens: int = 1,
) -> tuple[dict, dict] | None:
    """Run one inference pass and return (flat_inputs, flat_outputs).

    Returns None on failure so callers can skip that trial gracefully.
    """
    try:
        if is_enc_dec and hasattr(model, "encoder"):
            return _trace_encoder_decoder(model, inf_kwargs)

        # Try the decoder-only (with KV cache) path first
        if "input_ids" in inf_kwargs:
            result = _trace_decoder_only(model, inf_kwargs, fwd_params, n_new_tokens)
            if result is not None:
                return result

        # Fallback: single encoder-style forward
        return _trace_encoder_only(model, inf_kwargs)
    except Exception:
        return None


# ── dynamic-axis computation ──────────────────────────────────────────────────

def _compute_dynamic_axes(
    all_inputs: list[dict],
    all_outputs: list[dict],
) -> dict[str, dict[int, str]]:
    """Derive dynamic axes by comparing tensor shapes across multiple runs.

    A dimension is marked **dynamic** when its value differs between at least
    two runs, or when it is dimension 0 (batch) — which is always dynamic.
    """
    if len(all_inputs) < 2:
        # Only one run: fall back to "batch always dynamic, everything else static"
        axes: dict[str, dict[int, str]] = {}
        for name, shape in {**all_inputs[0], **all_outputs[0]}.items():
            s = _shape_of(shape)
            if s:
                axes[name] = {0: "batch"}
        return axes

    # Gather all tensor names seen across all runs
    all_names: set[str] = set()
    for inp, out in zip(all_inputs, all_outputs):
        all_names |= inp.keys() | out.keys()

    dynamic_axes: dict[str, dict[int, str]] = {}

    for name in all_names:
        shapes = []
        for inp, out in zip(all_inputs, all_outputs):
            raw = inp.get(name) or out.get(name)
            if raw is not None:
                shapes.append(_shape_of(raw))

        if not shapes:
            continue

        ndim = len(shapes[0])
        axes: dict[int, str] = {}
        for dim_idx in range(ndim):
            dim_values = {s[dim_idx] for s in shapes if len(s) > dim_idx}
            # Dynamic if batch dim (0) OR if the value actually varied
            if dim_idx == 0 or len(dim_values) > 1:
                label = "batch" if dim_idx == 0 else f"{name}_dim_{dim_idx}"
                axes[dim_idx] = label

        if axes:
            dynamic_axes[name] = axes

    return dynamic_axes


# ── public API ────────────────────────────────────────────────────────────────

def trace_model_shapes(
    model,
    inf_kwargs: dict[str, Any],
    skip_random_generation: bool = False,
    n_trials: int = 3,
) -> tuple[dict, dict, dict]:
    """Trace input/output shapes via multiple inference runs.

    Runs *n_trials* inferences (the first with the user-supplied *inf_kwargs*,
    the rest with randomly-generated shape variations) to empirically determine
    which tensor dimensions are truly dynamic rather than relying on config
    field heuristics.

    Args:
        model: PyTorch model to trace.
        inf_kwargs: The user-supplied inference keyword arguments (e.g. the
            tokenized input from the test file).  Used verbatim for trial 0
            and as the template shape for subsequent trials.
        skip_random_generation: When True the returned *inputs* dict stores
            the actual tensors rather than shape tuples (used as fixed dummy
            inputs during ONNX export).
        n_trials: Total number of inference passes.  More trials give more
            reliable dynamic-axis detection at the cost of extra latency.

    Returns:
        inputs      – {name: tensor_or_shape} for trial 0 (template for ONNX)
        outputs     – {name: shape_tuple} for trial 0
        dynamic_axes – {name: {dim_idx: axis_name}} derived from all trials
    """
    fwd_params = inspect.signature(model.forward).parameters
    is_enc_dec = getattr(getattr(model, "config", None), "is_encoder_decoder", False)

    # ── Trial 0: user-provided inputs ────────────────────────────────────────
    result0 = _trace_once(model, inf_kwargs, fwd_params, is_enc_dec, n_new_tokens=1)
    if result0 is None:
        raise RuntimeError("Initial inference trace failed — check that your "
                           "inference_kwargs are valid for this model.")

    all_inputs  = [result0[0]]
    all_outputs = [result0[1]]

    # ── Trials 1..n_trials-1: shape variations ───────────────────────────────
    # Also vary the decode query length (n_new_tokens) so the query
    # sequence-length dimension is correctly detected as dynamic.
    for trial_idx in range(1, n_trials):
        varied = _generate_variation(inf_kwargs, trial_idx, model)
        n_new = 1 + trial_idx          # 2, 3, 4, … new query tokens
        result_v = _trace_once(model, varied, fwd_params, is_enc_dec, n_new_tokens=n_new)
        if result_v is not None:
            all_inputs.append(result_v[0])
            all_outputs.append(result_v[1])

    # ── Derive dynamic axes from observed shape variation ─────────────────────
    dynamic_axes = _compute_dynamic_axes(all_inputs, all_outputs)

    # ── Build the returned inputs dict (trial 0 shapes / tensors) ────────────
    if skip_random_generation:
        # Store actual tensors for use as fixed ONNX dummy inputs
        returned_inputs = {
            k: inf_kwargs.get(k, torch.zeros(v, dtype=torch.long)
               if "id" in k or "mask" in k else torch.zeros(v))
            if not isinstance(v, tuple) else v
            for k, v in result0[0].items()
        }
    else:
        returned_inputs = result0[0]   # shape tuples

    return returned_inputs, result0[1], dynamic_axes
