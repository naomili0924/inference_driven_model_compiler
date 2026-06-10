from __future__ import annotations

import inspect
import os
import sys
import importlib.util
from typing import TYPE_CHECKING, Any, Callable

import torch


# ── Bring in symbols from the real optimum.exporters.onnx.utils ─────────────
# (things we don't override but convert.py needs)

def _real_onnx_utils_attr(name):
    """Load attribute from the site-packages optimum.exporters.onnx.utils."""
    _key = "_idmc_real_optimum_onnx_utils"
    if _key not in sys.modules:
        for _p in sys.path:
            if not _p or "inference_driven_model_compiler" in _p:
                continue
            _f = os.path.join(_p, "optimum", "exporters", "onnx", "utils.py")
            if os.path.exists(_f):
                spec = importlib.util.spec_from_file_location(_key, _f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[_key] = mod
                spec.loader.exec_module(mod)
                break
    return getattr(sys.modules.get(_key), name, None)


# Re-export from real package (needed by convert.py)
PickableInferenceSession = _real_onnx_utils_attr("PickableInferenceSession")
recursive_to_device = _real_onnx_utils_attr("recursive_to_device")

# The fallback submodel builder (used when inference tracing is not requested)
from optimum.exporters.utils import _get_submodels_and_export_configs


def _get_dummy_onnx_config():
    """Lazy import of DummyOnnxConfig to avoid circular dependency."""
    from optimum.exporters.onnx.model_configs import DummyOnnxConfig
    return DummyOnnxConfig


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

        # Scalar (0-d) tensors have no batch dimension to vary — e.g. a diffusion
        # UNet timestep passed as a bare scalar. Keep them verbatim.
        if not shape:
            varied[key] = val
            continue

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


def generate_config_dim(
    model: PreTrainedModel, 
    dim_name: list[str] | None = None,
):
    if dim_name is None:
        return {}
    tmp = {k: getattr(model.config, k) for k in dim_name if hasattr(model.config, k)}
    return {k: getattr(model.config, k) for k in dim_name if hasattr(model.config, k)}
    
def get_dynamic_models_for_export(
    pipeline,
    models_and_inputs: dict | None = None,
    models_and_outputs: dict | None = None,
    module_fixed_axis_fields: dict[str, list[str]] | None = None,
    int_dtype: str = "int64",
    float_dtype: str = "fp32"
):
    import copy
    import types
    from functools import partial
    DummyOnnxConfig = _get_dummy_onnx_config()

    models_for_export = {}
    text_encoder = pipeline.text_encoder
    text_encoder_config = DummyOnnxConfig(config=text_encoder.config,
                                          task="text-encoding", 
                                          preprocessors=None, 
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["text_encoder"],
                                          model_outputs=models_and_outputs["text_encoder"],
                                          config_dim=generate_config_dim(text_encoder, module_fixed_axis_fields["text_encoder"]))
    models_for_export["text_encoder"] = (text_encoder, text_encoder_config)

    if hasattr(pipeline, "text_encoder_2") and "text_encoder_2" in models_and_outputs.keys():
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_2_config = DummyOnnxConfig(config=text_encoder_2.config, 
                                                task="text-encoding", 
                                                preprocessors=None, 
                                                int_dtype=int_dtype,
                                                float_dtype=float_dtype,
                                                model_inputs=models_and_inputs["text_encoder_2"],
                                                model_outputs=models_and_outputs["text_encoder_2"],
                                                config_dim=generate_config_dim(text_encoder, module_fixed_axis_fields["text_encoder_2"]))
        models_for_export["text_encoder_2"] = (text_encoder_2, text_encoder_2_config) 

    transformer = pipeline.transformer
    transformer_config = DummyOnnxConfig(config=transformer.config, 
                                          task="backbone", 
                                          preprocessors=None, 
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["transformer"],
                                          model_outputs=models_and_outputs["transformer"],
                                          config_dim=generate_config_dim(transformer, module_fixed_axis_fields["transformer"]))
    models_for_export["transformer"] = (transformer, transformer_config)

    if "vae_encoder" in models_and_inputs.keys():
        vae_encoder = copy.deepcopy(pipeline.vae)
        # proper forward wrapper
        def encode_forward(self, sample):
            return vae_encoder.encode(self, x=sample, return_dict=False)
        vae_encoder.forward = types.MethodType(encode_forward, vae_encoder)
        vae_encoder_config = DummyOnnxConfig(config=vae_encoder.config, 
                                              task="sample_encode", 
                                              preprocessors=None, 
                                              int_dtype=int_dtype,
                                              float_dtype=float_dtype,
                                              model_inputs=models_and_inputs["vae_encoder"],
                                              model_outputs=models_and_outputs["vae_encoder"],
                                              config_dim=generate_config_dim(vae_decoder, module_fixed_axis_fields["vae_encoder"]))
        models_for_export["vae_encoder"] = (vae_encoder, vae_encoder_config)

    if "vae_decoder" in models_and_inputs.keys():
        vae_decoder = copy.deepcopy(pipeline.vae)
        # proper forward wrapper
        def decode_forward(self, latent_sample):
            return vae_decoder.decode(self, z=latent_sample, return_dict=False)
        vae_decoder.forward = types.MethodType(decode_forward, vae_decoder)
        vae_decoder_config = DummyOnnxConfig(config=vae_decoder.config, 
                                              task="latent_decode", 
                                              preprocessors=None, 
                                              int_dtype=int_dtype,
                                              float_dtype=float_dtype,
                                              model_inputs=models_and_inputs["vae_decoder"],
                                              model_outputs=models_and_outputs["vae_decoder"],
                                              config_dim=generate_config_dim(vae_decoder, module_fixed_axis_fields["vae_decoder"]))
        models_for_export["vae_decoder"] = (vae_decoder, vae_decoder_config)
    return models_for_export


def get_dynamic_model_for_export(
    model,
    models_and_inputs: dict | None = None,
    models_and_outputs: dict | None = None,
    module_fixed_axis_fields: dict[str, list[str]] | None = None,
    int_dtype: str = "int64",
    float_dtype: str = "fp32"
):
    DummyOnnxConfig = _get_dummy_onnx_config()

    # For encoder-decoder models we traced (and will export) only the encoder.
    # Exporting the full model with encoder-only inputs causes the decoder to fail.
    is_enc_dec = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
    export_model = model
    if is_enc_dec:
        _enc = getattr(model, "encoder", None) or (
            model.get_encoder() if hasattr(model, "get_encoder") else None
        )
        if _enc is not None:
            export_model = _enc

    transformer_config = DummyOnnxConfig(config=model.config,
                                          task="backbone",
                                          preprocessors=None,
                                          int_dtype=int_dtype,
                                          float_dtype=float_dtype,
                                          model_inputs=models_and_inputs["transformer"],
                                          model_outputs=models_and_outputs["transformer"],
                                          config_dim=generate_config_dim(model, (module_fixed_axis_fields or {}).get("transformer", [])))
    models_for_export = {}
    models_for_export["transformer"] = (export_model, transformer_config)
    return models_for_export
    

def _get_submodels_and_onnx_configs(
    model: PreTrainedModel,
    task: str,
    monolith: bool,
    custom_onnx_configs: dict,
    custom_architecture: bool,
    _variant: str,
    library_name: str,
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
    fn_get_submodels: Callable | None = None,
    preprocessors: list[Any] | None = None,
    model_kwargs: dict | None = None,
    models_and_inputs: dict | None = None,
    models_and_outputs: dict | None = None,
    module_fixed_axis_fields: dict[str, list[str]] | None = None,
):
    if library_name == "transformers" and model.config.model_type == "metaclip_2":
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="onnx", task=task, library_name="transformers"
        )
        export_config = export_config_constructor(
            model.config,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            preprocessors=preprocessors,
        )
        export_config.variant = _variant
        return export_config, get_metaclip_2_models_for_export(model, export_config)

    if library_name == "diffusers" and model.__class__.__name__.startswith("Sana"):
        return None, get_sana_models_for_export(model, int_dtype, float_dtype)

    ## use inference to trace input and output shape
    if library_name == "diffusers" and models_and_inputs is not None and models_and_outputs is not None and module_fixed_axis_fields is not None:
        return None, get_dynamic_models_for_export(model, models_and_inputs, models_and_outputs, module_fixed_axis_fields, int_dtype, float_dtype)

    if library_name == "transformers" and models_and_inputs is not None and models_and_outputs is not None and module_fixed_axis_fields is not None:
        onnx_config = get_dynamic_model_for_export(model, models_and_inputs, models_and_outputs, module_fixed_axis_fields, int_dtype, float_dtype)
        return onnx_config["transformer"][1], onnx_config

    return _get_submodels_and_export_configs(
        model,
        task,
        monolith,
        custom_onnx_configs,
        custom_architecture,
        _variant,
        library_name,
        int_dtype,
        float_dtype,
        fn_get_submodels,
        preprocessors,
        model_kwargs,
        exporter="onnx",
    )

def make_positional_hook(dummy_inputs, module_name):
    import inspect
    def hook(module, args, kwargs):
        sig = inspect.signature(module.forward)
        params = list(sig.parameters.values())
        # remove self if present
        if params and params[0].name == "self":
            params = params[1:]
        named_shapes = {}
        for p, v in zip(params, args):
            if torch.is_tensor(v):
                named_shapes[p.name] = tuple(v.shape)
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                named_shapes[k] = tuple(v.shape)
        dummy_inputs[module_name] = named_shapes
        return None  # do not modify inputs
    return hook

def get_output_name_and_shape(output, name):
    from dataclasses import fields, is_dataclass

    named_shapes = {}
    if torch.is_tensor(output):
        named_shapes[name] = tuple(output.shape)
    elif is_dataclass(output):
        for f in fields(output):
            val = getattr(output, f.name)
            if torch.is_tensor(val):
                named_shapes[f.name] = tuple(val.shape)
    elif isinstance(output, (tuple, list)):
        for i, x in enumerate(output):
            if torch.is_tensor(x):
                named_shapes[f"{name}_{i}"] = tuple(x.shape)
    elif isinstance(output, dict):
        for k, v in output.items():
            if torch.is_tensor(v):
                named_shapes[k] = tuple(v.shape)
    return named_shapes
    

def make_dataclass_output_hook(dummy_outputs, module_name):
    def hook(module, args, output):
        dummy_outputs[module_name] = get_output_name_and_shape(output, "sample")
        return None  # don't modify output
    return hook

def _infer_transformer_kwargs(model) -> dict:
    """Auto-generate inference kwargs for a PreTrainedModel using its tokenizer.

    Tries (in order):
    1. Load the tokenizer from the model's name/path and encode a dummy sentence.
    2. Inspect the model's forward signature and generate random tensors for
       required tensor-typed parameters.

    After either attempt, supplements with modality-specific inputs that the
    tokenizer alone cannot produce (e.g. `input_features` for audio models like
    Whisper).
    """
    import inspect
    model_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    result = {}

    # --- Attempt 1: use the tokenizer ---
    if model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
            result = dict(inputs)
        except Exception:
            pass

    # --- Supplement: audio-primary models (Whisper, etc.) need input_features ---
    # Only add when input_ids is absent from the forward signature: that means
    # the model is audio-driven (Whisper). Multimodal models like Gemma 4 also
    # have input_features in their signature but are text-primary (input_ids is
    # their required argument) — don't inject audio tensors for those.
    fwd_params = inspect.signature(model.forward).parameters
    if ("input_features" in fwd_params
            and "input_features" not in result
            and "input_ids" not in fwd_params):
        cfg = getattr(model, "config", None)
        num_mel_bins = getattr(cfg, "num_mel_bins", 80)
        max_src_pos  = getattr(cfg, "max_source_positions", 1500)
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        result["input_features"] = torch.zeros(
            (1, num_mel_bins, max_src_pos * 2), dtype=model_dtype
        )

    if result:
        return result

    # --- Attempt 2: inspect forward signature (text models without a tokenizer) ---
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", 32000)
    for name, param in fwd_params.items():
        if name in ("self", "return_dict", "output_attentions",
                    "output_hidden_states", "labels"):
            continue
        if param.default is not inspect.Parameter.empty and param.default is None:
            continue  # skip optional tensor params
        if "ids" in name or "tokens" in name:
            result[name] = torch.randint(0, vocab_size, (1, 16))
        elif "mask" in name:
            result[name] = torch.ones((1, 16), dtype=torch.long)
        elif "type" in name:
            result[name] = torch.zeros((1, 16), dtype=torch.long)
    return result


def _get_submodels_and_tensors_(
    model: PreTrainedModel | DiffusionPipeline,
    inf_kwargs: dict[str, Any] | None = None,
    skip_random_generation: bool = False,
    use_cache: bool = False,
):
    from transformers import PreTrainedModel
    if isinstance(model, PreTrainedModel):
        dummy_inputs = {"transformer": {}}
        dummy_outputs = {"transformer": {}}

        # Auto-infer inf_kwargs from the tokenizer when not provided
        if inf_kwargs is None:
            inf_kwargs = _infer_transformer_kwargs(model)

        # Add position_ids derived from input_ids if not already present
        import inspect as _inspect
        _fwd_params = set(_inspect.signature(model.forward).parameters)
        prefill_kwargs = dict(inf_kwargs)
        if "input_ids" in prefill_kwargs:
            input_ids = prefill_kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            if "position_ids" not in prefill_kwargs:
                prefill_kwargs["position_ids"] = torch.arange(
                    seq_len, dtype=torch.long
                ).unsqueeze(0).expand(batch_size, -1)
        else:
            batch_size, seq_len = 1, 16

        # Ensure KV-cache is enabled for the prefill pass when requested
        if use_cache and "use_cache" not in prefill_kwargs:
            prefill_kwargs["use_cache"] = True

        # Encoder-decoder models (T5, BART, Whisper, …): trace the encoder
        # submodule directly.  Running the full model forward requires both
        # encoder and decoder inputs and is not needed for encoder-only export.
        is_enc_dec = getattr(getattr(model, "config", None), "is_encoder_decoder", False)
        _encoder_mod = (
            getattr(model, "encoder", None)
            or (model.get_encoder() if hasattr(model, "get_encoder") else None)
        )
        if is_enc_dec and _encoder_mod is not None:
            import inspect as _enc_inspect
            enc_sig = set(_enc_inspect.signature(_encoder_mod.forward).parameters)
            enc_kwargs = {
                k: v for k, v in prefill_kwargs.items()
                if k in enc_sig and torch.is_tensor(v)
            }
            with torch.no_grad():
                enc_out = _encoder_mod(**enc_kwargs)
            for key, val in enc_kwargs.items():
                dummy_inputs["transformer"][key] = (
                    val if skip_random_generation else tuple(val.shape)
                )
            dummy_outputs["transformer"].update(_flatten_output(enc_out))
            return dummy_inputs, dummy_outputs

        # Prefill forward: no past_key_values, traces all input shapes
        with torch.no_grad():
            prefill_output = model(**prefill_kwargs)

        past_key_values = getattr(prefill_output, "past_key_values", None)

        def _iter_pkv(pkv):
            """Yield (layer_idx, key_tensor, value_tensor) from any cache format."""
            if pkv is None:
                return
            if hasattr(pkv, "layers"):
                # transformers 5.x: DynamicCache / HybridCache with .layers list
                for i, layer in enumerate(pkv.layers):
                    k = getattr(layer, "keys", None)
                    v = getattr(layer, "values", None)
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        yield i, k, v
            elif hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                # transformers 4.38-4.x DynamicCache with .key_cache / .value_cache
                for i, (k, v) in enumerate(zip(pkv.key_cache, pkv.value_cache)):
                    if torch.is_tensor(k) and torch.is_tensor(v):
                        yield i, k, v
            else:
                # Legacy tuple-of-tuples: ((k0, v0), (k1, v1), ...)
                for i, entry in enumerate(pkv):
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        k, v = entry
                        if torch.is_tensor(k) and torch.is_tensor(v):
                            yield i, k, v

        if past_key_values is not None and len(past_key_values) > 0:
            # --- Decode step: trace "with past" scenario ---
            input_ids = prefill_kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape

            # Single new token
            new_token = torch.zeros((batch_size, 1), dtype=input_ids.dtype)
            # Extended attention mask (original seq + 1 new token)
            new_attn_mask = torch.ones((batch_size, seq_len + 1), dtype=torch.long)
            # Position of the new token
            new_pos_ids = torch.full((batch_size, 1), seq_len, dtype=torch.long)

            decode_kwargs = {
                "input_ids": new_token,
                "attention_mask": new_attn_mask,
                "position_ids": new_pos_ids,
                "past_key_values": past_key_values,
            }
            if use_cache:
                decode_kwargs["use_cache"] = True
            # cache_position: required by some models during tracing
            if "cache_position" in _fwd_params:
                decode_kwargs["cache_position"] = torch.tensor([seq_len], dtype=torch.long)

            with torch.no_grad():
                decode_output = model(**decode_kwargs)

            def _store(d, key, val):
                d[key] = val if skip_random_generation else tuple(val.shape)

            # Record flat tensor inputs (not past_key_values yet)
            for key, val in decode_kwargs.items():
                if torch.is_tensor(val):
                    _store(dummy_inputs["transformer"], key, val)

            # Flatten past_key_values into named inputs.
            for i, k, v in _iter_pkv(decode_kwargs["past_key_values"]):
                _store(dummy_inputs["transformer"], f"past_key_values.{i}.key", k)
                _store(dummy_inputs["transformer"], f"past_key_values.{i}.value", v)

            # Record outputs
            if getattr(decode_output, "logits", None) is not None:
                dummy_outputs["transformer"]["logits"] = tuple(decode_output.logits.shape)
            updated_pkv = getattr(decode_output, "past_key_values", None)
            if updated_pkv is not None:
                # ONNX convention: KV-cache *inputs* are named "past_key_values.*"
                # while the updated KV-cache *outputs* are named "present.*". Using
                # distinct names avoids an input/output name collision (which makes
                # torch.onnx append a ".1" suffix to the inputs) and matches what
                # optimum's validation expects (it renames the PyTorch reference's
                # "past_key_values" output to "present" before comparing).
                for i, k, v in _iter_pkv(updated_pkv):
                    dummy_outputs["transformer"][f"present.{i}.key"] = tuple(k.shape)
                    dummy_outputs["transformer"][f"present.{i}.value"] = tuple(v.shape)
        else:
            # No KV cache: original single-step behaviour + position_ids
            for key, val in prefill_kwargs.items():
                dummy_inputs["transformer"][key] = (
                    val if skip_random_generation else tuple(val.shape)
                )
            hooks = [model.register_forward_hook(
                make_dataclass_output_hook(dummy_outputs, "transformer")
            )]
            model(**prefill_kwargs)
            for h in hooks:
                h.remove()

        return dummy_inputs, dummy_outputs
        
        
    import torch.nn as nn
    import inspect
    import types
    
    # key: module_name, value: {input_name: tensor_shape}
    dummy_inputs = {}
    dummy_outputs = {}

    hooks = []
    transformer_original_forward = None
    orig_decode = None
    orig_encode = None

    for name, module in model.components.items():
        if isinstance(module, nn.Module):
            dummy_inputs[name] = {}
            dummy_outputs[name] = {}

    if "text_encoder" in dummy_inputs.keys():
        hooks.append(
            model.text_encoder.register_forward_pre_hook(make_positional_hook(dummy_inputs, "text_encoder"), with_kwargs=True))
        hooks.append(
            model.text_encoder.register_forward_hook(make_dataclass_output_hook(dummy_outputs, "text_encoder")))

    if "text_encoder_2" in dummy_inputs.keys():
        hooks.append(
            model.text_encoder_2.register_forward_pre_hook(make_positional_hook(dummy_inputs, "text_encoder_2"), with_kwargs=True))
        hooks.append(
            model.text_encoder_2.register_forward_hook(make_dataclass_output_hook(dummy_outputs, "text_encoder_2")))

    if "transformer" in dummy_inputs.keys():
        transformer_original_forward = model.transformer.forward
        def wrapped_forward(*args, **kwargs):
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    dummy_inputs["transformer"][key] = tuple(value.shape)
            return transformer_original_forward(*args, **kwargs)
        
        model.transformer.forward = wrapped_forward
        hooks.append(
            model.transformer.register_forward_hook(make_dataclass_output_hook(dummy_outputs, "transformer")))

    if "vae" in dummy_inputs.keys():
        dummy_inputs["vae_encoder"] = {}
        dummy_inputs["vae_decoder"] = {}
        # hook encoder
        wrap_encode = model.vae.encode
        for cell in wrap_encode.__closure__:
            if inspect.isfunction(cell.cell_contents):
                orig_decode = cell.cell_contents
                break
        if orig_encode is None:
            sig = None
        else:
            sig = inspect.signature(orig_encode)
        def hooked_encode(self, *args, **kwargs):
            if sig is not None:
                bound = sig.bind_partial(self, *args, **kwargs)
                for name, value in bound.arguments.items():
                    if torch.is_tensor(value):
                        dummy_inputs["vae_encoder"][name] = tuple(value.shape)
            output = wrap_encode(*args, **kwargs)
            dummy_output["vae_encoder"] = get_output_name_and_shape(output, "latent_dist")
            return output
        model.vae.encode = types.MethodType(hooked_encode, model.vae)

        wrap_decode = model.vae.decode
        for cell in wrap_decode.__closure__:
            if inspect.isfunction(cell.cell_contents):
                orig_decode = cell.cell_contents
                break
        if orig_decode is None:
            sig = None
        else:
            sig = inspect.signature(orig_decode)
        def hooked_decode(self, *args, **kwargs):
            if sig is not None:
                bound = sig.bind_partial(self, *args, **kwargs)
                for name, value in bound.arguments.items():
                    if torch.is_tensor(value):
                        dummy_inputs["vae_decoder"]["latent_sample"] = tuple(value.shape)
            output = wrap_decode(*args, **kwargs)
            dummy_outputs["vae_decoder"] = get_output_name_and_shape(output, "sample")
            return output
        model.vae.decode = types.MethodType(hooked_decode, model.vae)

    output = model(**inf_kwargs).frames[0]  # yes, we can inference

    filtered_inputs = {k: v for k, v in dummy_inputs.items() if v}
    filtered_outputs = {k: v for k, v in dummy_outputs.items() if v}

    # remove all the model hooks 
    for h in hooks:
        h.remove()
    if transformer_original_forward is not None:
        model.transformer.forward = transformer_original_forward
    if orig_decode is not None:
        model.vae.decode = orig_decode
    if orig_encode is not None:
        model.vae.encode = orig_encode
    
    return filtered_inputs, filtered_outputs
    