from __future__ import annotations

import importlib
import inspect
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils.constants import CONFIG_NAME
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.utils import http_user_agent

from onnxruntime import InferenceSession, SessionOptions
from optimum.exporters.onnx import main_export
from optimum.onnxruntime.base import ORTParentMixin, ORTSessionMixin
from optimum.onnxruntime.utils import get_device_for_provider, prepare_providers_and_provider_options
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
    DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
    ONNX_WEIGHTS_NAME,
    is_diffusers_version,
)

from .utils import load_shapes_as_torch_size

if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution  # type: ignore

if is_diffusers_version(">=", "0.35.0"):
    from diffusers.models.cache_utils import CacheMixin
else:
    CacheMixin = object

logger = logging.getLogger(__name__)


class ORTModelMixin(ORTSessionMixin, ConfigMixin, CacheMixin):
    config_name: str = CONFIG_NAME

    def __init__(
        self,
        session: InferenceSession,
        parent: "ORTDiffusionPipeline",
        use_io_binding: bool | None = None,
    ):
        self.initialize_ort_attributes(session, use_io_binding=use_io_binding)
        self.parent = parent

        config_file_path = Path(session._model_path).parent / self.config_name
        if not config_file_path.is_file():
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")
        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)

        self.io_binding_file = None

    def set_io_binding_file(self, filename: str):
        self.io_binding_file = filename

    def save_pretrained(self, save_directory: str | Path):
        self.save_session(save_directory)
        self.save_config(save_directory)

    def named_modules(self):
        # diffusers >= 0.35.0 calls named_modules for KV-cache hooks — not applicable to ORT
        yield from []


class ORTUnet(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "time_cond_proj_dim"):
            self.register_to_config(time_cond_proj_dim=None)

        if len(self.input_shapes["timestep"]) > 0:
            logger.warning(
                "The exported unet onnx model expects a non-scalar timestep input. "
                "Please re-export the pipeline with a newer version of optimum and diffusers."
            )

    def forward(
        self,
        sample: np.ndarray | torch.Tensor,
        timestep: np.ndarray | torch.Tensor,
        encoder_hidden_states: np.ndarray | torch.Tensor,
        timestep_cond: np.ndarray | torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        if len(self.input_shapes["timestep"]) > 0:
            timestep = timestep.unsqueeze(0)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep_cond": timestep_cond,
            **(cross_attention_kwargs or {}),
            **(added_cond_kwargs or {}),
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            known_output_shapes["out_sample"] = sample.shape
            known_output_buffers = None
            if "LatentConsistencyModel" not in self.parent.__class__.__name__:
                known_output_buffers = {"out_sample": sample}
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                known_output_shapes=known_output_shapes,
                known_output_buffers=known_output_buffers,
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        model_outputs["sample"] = model_outputs.pop("out_sample")

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTTransformer(ORTModelMixin):
    def forward(
        self,
        hidden_states: np.ndarray | torch.Tensor,
        encoder_hidden_states: np.ndarray | torch.Tensor,
        timestep: np.ndarray | torch.Tensor,
        pooled_projections: np.ndarray | torch.Tensor | None = None,
        guidance: np.ndarray | torch.Tensor | None = None,
        txt_ids: np.ndarray | torch.Tensor | None = None,
        img_ids: np.ndarray | torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        # CogVideoX-specific (constant-folded into ONNX, accepted but not forwarded)
        timestep_cond: np.ndarray | torch.Tensor | None = None,
        ofs: np.ndarray | torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(hidden_states, torch.Tensor)

        model_inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "guidance": guidance,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            **(joint_attention_kwargs or {}),
            **(attention_kwargs or {}),
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            known_output_shapes["out_hidden_states"] = hidden_states.shape
            known_output_buffers = None
            if "Flux" not in self.parent.__class__.__name__:
                known_output_buffers = {"out_hidden_states": hidden_states}
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                known_output_shapes=known_output_shapes,
                known_output_buffers=known_output_buffers,
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTTextEncoder(ORTModelMixin):
    def forward(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes, known_output_buffers=None
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))
        else:
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs.pop(f"hidden_states.{i}", None)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVaeEncoder(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.config, "scaling_factor"):
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        sample: np.ndarray | torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(sample, torch.Tensor)
        model_inputs = {"sample": sample}

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes, known_output_buffers=None
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")
        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVaeDecoder(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.config, "scaling_factor") and hasattr(self.config, "block_out_channels"):
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))
        self._cpu_session = None  # lazily created if GPU OOM occurs

    def forward(
        self,
        latent_sample: np.ndarray | torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)
        # The ONNX model's first input may be named "latent_sample" (standard) or
        # the raw forward-argument name like "x" (inference-driven export from WanDecoder).
        # Use the actual session input name so _prepare_io_binding can find it.
        # input_names is a {name: idx} dict, so use next(iter(...)) for the first key.
        actual_input_name = next(iter(self.input_names)) if self.input_names else "latent_sample"
        model_inputs = {actual_input_name: latent_sample}

        # VAE decoder: try GPU session first; on OOM fall back to CPU session.
        # The full-video upsample intermediates can exceed the CUDA BFC arena limit.
        import onnxruntime as ort
        onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
        try:
            onnx_outputs = self.session.run(None, onnx_inputs)
        except Exception as gpu_err:
            if "allocate memory" not in str(gpu_err) and "RuntimeException" not in str(gpu_err):
                raise
            # GPU OOM — create / reuse a CPU session
            if self._cpu_session is None:
                logger.warning(
                    "VAE decoder GPU OOM — falling back to CPU execution for decode step."
                )
                self._cpu_session = ort.InferenceSession(
                    self.session._model_path,
                    providers=["CPUExecutionProvider"],
                )
            cpu_inputs = {
                k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in onnx_inputs.items()
            }
            onnx_outputs = self._cpu_session.run(None, cpu_inputs)
        model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVae(ORTParentMixin):
    def __init__(self, encoder: ORTVaeEncoder | None = None, decoder: ORTVaeDecoder | None = None):
        self.encoder = encoder
        self.decoder = decoder
        self.initialize_ort_attributes(parts=list(filter(None, {self.encoder, self.decoder})))

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @property
    def config(self):
        return self.decoder.config


def _register_onnx_upsample_symbolics():
    """Register custom ONNX symbolic for aten::_upsample_nearest_exact2d.

    PyTorch's nearest-exact upsampling has no built-in ONNX opset-18 symbolic.
    Map it to the ONNX Resize op (nearest mode, half_pixel coordinates).
    The JIT op signature is: (input, output_size, scale_factors[h, w])
    so scale_h here is the [h_scale, w_scale] float[] constant.
    """
    try:
        from torch.onnx import symbolic_helper

        @symbolic_helper.parse_args("v", "v", "v")
        def _upsample_nearest_exact2d_sym(g, input, output_size, scale_h=None):
            # scale_h = [h_scale, w_scale]; prepend [1.0, 1.0] for batch + channel.
            ones = g.op("Constant", value_t=torch.tensor([1.0, 1.0], dtype=torch.float32))
            scales = g.op("Concat", ones, scale_h, axis_i=0)
            empty_roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
            return g.op(
                "Resize", input, empty_roi, scales,
                mode_s="nearest",
                coordinate_transformation_mode_s="half_pixel",
                nearest_mode_s="round_prefer_floor",
            )

        torch.onnx.register_custom_op_symbolic(
            "aten::_upsample_nearest_exact2d",
            _upsample_nearest_exact2d_sym,
            18,
        )
    except Exception as exc:
        logger.warning(f"Could not register _upsample_nearest_exact2d symbolic: {exc}")


def _on_the_fly_diffusion_export(
    model_name_or_path,
    output,
    inference_kwargs,
    module_fixed_axis_fields=None,
    torch_dtype=None,
    device="cpu",
    hub_kwargs=None,
    n_trials=3,
    skip_random_generation=False,
):
    """Load the PyTorch diffusion pipeline, trace per-submodule tensor shapes
    via forward hooks on one inference pass, then export each submodule to ONNX.
    """
    import json
    import inspect
    from types import SimpleNamespace

    from inference_driven_model_compiler.optimum.exporters.onnx.utils import (
        trace_model_shapes,
        generate_config_dim,
    )
    from inference_driven_model_compiler.optimum.exporters.onnx.model_configs import DummyOnnxConfig

    _register_onnx_upsample_symbolics()

    # 1. Load the PyTorch pipeline
    load_kw = {**(hub_kwargs or {})}
    if torch_dtype is not None:
        load_kw["torch_dtype"] = torch_dtype
    pt_pipeline = DiffusionPipeline.from_pretrained(str(model_name_or_path), **load_kw)
    pt_pipeline = pt_pipeline.to(device)

    # 2. Collect submodules to export: (name, module, subfolder, config_source)
    vae = getattr(pt_pipeline, "vae", None)
    specs = []
    _vae_decode_export_overrides: dict[str, torch.nn.Module] = {}
    for name, subfolder in [
        ("text_encoder",   DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER),
        ("text_encoder_2", DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER),
        ("text_encoder_3", DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER),
        ("transformer",    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER),
        ("unet",           DIFFUSION_MODEL_UNET_SUBFOLDER),
    ]:
        mod = getattr(pt_pipeline, name, None)
        if isinstance(mod, torch.nn.Module):
            specs.append((name, mod, subfolder, mod))
    if vae is not None:
        for vae_name, vae_attr, vae_subfolder in [
            ("vae_encoder", "encoder", DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER),
        ]:
            sub = getattr(vae, vae_attr, None)
            if isinstance(sub, torch.nn.Module):
                specs.append((vae_name, sub, vae_subfolder, vae))

        # For vae_decoder: hook post_quant_conv to capture the full latent z,
        # then export a wrapper (post_quant_conv → decoder) that takes the full
        # latent and decodes it in one shot (no frame-by-frame caching in ONNX).
        _pqc = getattr(vae, "post_quant_conv", None)
        _dec = getattr(vae, "decoder", None)
        if _pqc is not None and _dec is not None:
            # WAN-style: post_quant_conv is called ONCE on the full latent.
            # Hook pqc to capture the full z; export pqc+decoder together.
            class _VaeFullDecodeWrapper(torch.nn.Module):
                def __init__(self, pqc, dec):
                    super().__init__()
                    self.pqc = pqc
                    self.dec = dec

                def forward(self, x):
                    out = self.dec(self.pqc(x))
                    # Some decoders (e.g. CogVideoX) return (tensor, conv_cache_dict).
                    # Only the decoded tensor is needed for ONNX.
                    return out[0] if isinstance(out, (list, tuple)) else out

            _vae_full_decode = _VaeFullDecodeWrapper(_pqc, _dec)
            specs.append(("vae_decoder", _pqc, DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, vae))
            _vae_decode_export_overrides = {"vae_decoder": _vae_full_decode}

        elif _dec is not None:
            # CogVideoX-style: post_quant_conv is None; decoder is called directly
            # per-frame-batch with the raw latent slice. Hook the decoder to capture
            # a representative slice, then export a wrapper that accepts z and returns
            # only the decoded tensor (dropping conv_cache).
            class _VaeNoQuantDecodeWrapper(torch.nn.Module):
                def __init__(self, dec):
                    super().__init__()
                    self.dec = dec

                def forward(self, z):
                    out = self.dec(z)
                    return out[0] if isinstance(out, (list, tuple)) else out

            _vae_full_decode = _VaeNoQuantDecodeWrapper(_dec)
            specs.append(("vae_decoder", _dec, DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER, vae))
            _vae_decode_export_overrides = {"vae_decoder": _vae_full_decode}
        else:
            _vae_decode_export_overrides = {}

    # 3. Register forward pre-hooks to capture each submodule's first-call inputs.
    # We snapshot (deep-copy) values immediately — some models (e.g. WanDecoder3d)
    # pass mutable lists (feat_cache, feat_idx) that are mutated in-place during
    # the forward pass, so a plain reference would be stale by the time we use it.
    import copy

    captured = {}
    hooks = []
    for name, mod, _, _ in specs:
        fwd_params = list(inspect.signature(mod.forward).parameters.keys())
        fwd_param_set = set(fwd_params)

        def make_hook(mod_name, param_list, param_set):
            def hook(module, args, kwargs_fwd):
                if mod_name in captured:
                    return

                def _snap(v):
                    if torch.is_tensor(v):
                        return v.detach().clone()
                    if isinstance(v, (list, tuple, dict)):
                        return copy.deepcopy(v)
                    return v

                bound = {}
                for i, arg in enumerate(args):
                    if i < len(param_list):
                        bound[param_list[i]] = _snap(arg)
                if kwargs_fwd:
                    for k, v in kwargs_fwd.items():
                        if k in param_set:
                            bound[k] = _snap(v)
                captured[mod_name] = bound
            return hook

        h = mod.register_forward_pre_hook(
            make_hook(name, fwd_params, fwd_param_set), with_kwargs=True
        )
        hooks.append(h)

    # 4. Run one full pipeline inference pass — hooks capture submodule inputs
    with torch.no_grad():
        pt_pipeline(**inference_kwargs)

    for h in hooks:
        h.remove()

    # Move each captured submodule to CPU so ONNX export dummy inputs (always CPU)
    # match the model device, and so shape-variation trials don't hit device mismatches.
    def _to_cpu(v):
        if torch.is_tensor(v):
            return v.cpu()
        if isinstance(v, list):
            return [_to_cpu(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_to_cpu(x) for x in v)
        if isinstance(v, dict):
            return {k2: _to_cpu(v2) for k2, v2 in v.items()}
        return v

    for name, mod, _, _ in specs:
        if name in captured:
            mod.cpu()
            captured[name] = {k: _to_cpu(v) for k, v in captured[name].items()}
    # Also move the full vae_decoder wrapper (which includes pqc + decoder) to CPU
    for _, wrapper_mod in _vae_decode_export_overrides.items():
        wrapper_mod.cpu()

    # Normalize the vae_decoder captured input key to "z" when exported via
    # _VaeNoQuantDecodeWrapper (CogVideoX-style, no post_quant_conv).  The hook
    # names the key after the decoder's first parameter ("sample" for CogVideoX),
    # but the wrapper's forward signature uses "z" so trace_model_shapes must
    # receive {"z": tensor}.
    if "vae_decoder" in captured and "vae_decoder" in _vae_decode_export_overrides:
        wrapper = _vae_decode_export_overrides["vae_decoder"]
        if hasattr(wrapper, "dec") and not hasattr(wrapper, "pqc"):
            # _VaeNoQuantDecodeWrapper — rename first captured tensor key to "z"
            cap = captured["vae_decoder"]
            tensor_keys = [k for k, v in cap.items() if torch.is_tensor(v)]
            if tensor_keys and tensor_keys[0] != "z":
                cap["z"] = cap.pop(tensor_keys[0])
                captured["vae_decoder"] = cap

    # 5. Trace shapes and build ONNX configs for each captured submodule
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    models_and_onnx_configs = {}
    ordered_names = []
    subfolder_map = {}
    io_binding_outputs = {}

    for name, mod, subfolder, config_src in specs:
        if name not in captured:
            logger.warning(f"Submodule '{name}' was not called during inference — skipping ONNX export.")
            continue

        # Pass only tensor inputs to trace_model_shapes — non-tensor arguments
        # (e.g. feat_cache, feat_idx, first_chunk, return_dict) are internal state
        # that don't belong in the ONNX interface and would cause mutable-list
        # mutation issues across multiple tracing trials.
        tensor_inputs = {k: v for k, v in captured[name].items() if torch.is_tensor(v)}
        if not tensor_inputs:
            logger.warning(f"Submodule '{name}' has no tensor inputs — skipping ONNX export.")
            continue
        # For vae_decoder: trace the full wrapper (pqc + decoder) using the
        # captured pqc inputs, so the output shapes reflect the decoded video.
        trace_mod = _vae_decode_export_overrides.get(name, mod)
        try:
            inputs, outputs, dynamic_axes = trace_model_shapes(
                trace_mod,
                tensor_inputs,
                skip_random_generation=skip_random_generation,
                n_trials=n_trials,
            )
        except Exception as exc:
            import traceback
            logger.warning(
                f"Shape tracing failed for '{name}': {exc}\n{traceback.format_exc()} — skipping."
            )
            continue

        # For vae_decoder the captured input is a single chunk (e.g. 3 frames), but at
        # inference the full temporal extent varies.  Force dim-2 (frames) to be dynamic
        # so the exported ONNX accepts any number of frames.
        if name == "vae_decoder":
            for inp_name in list(dynamic_axes.keys()):
                dynamic_axes[inp_name][2] = "num_frames"

        dim_names = (module_fixed_axis_fields or {}).get(name, [])
        cfg = (
            getattr(config_src, "config", None)
            or getattr(mod, "config", None)
            or SimpleNamespace(model_type="placeholder")
        )
        # For vae_decoder/vae_encoder, mod has no .config — use config_src (the VAE)
        config_dim_src = config_src if hasattr(config_src, "config") else mod
        config_dim = generate_config_dim(config_dim_src, dim_names)

        # Detect float dtype from the captured inputs (actual tensors from inference),
        # not from model parameters (which may be mixed fp16/fp32 across layers).
        float_dtype = "fp32"
        for _v in captured[name].values():
            if torch.is_tensor(_v) and _v.is_floating_point():
                if _v.dtype == torch.float16:
                    float_dtype = "fp16"
                elif _v.dtype == torch.bfloat16:
                    float_dtype = "bf16"
                break

        onnx_cfg = DummyOnnxConfig(
            config=cfg,
            task="backbone",
            model_inputs=inputs,
            model_outputs=outputs,
            config_dim=config_dim,
            dynamic_axes=dynamic_axes,
            float_dtype=float_dtype,
        )

        # export_pytorch does `model.config.return_dict = True` unconditionally.
        # Sub-modules (e.g. WanDecoder3d) have no .config — attach a fake one.
        if not hasattr(mod, "config"):
            mod.config = cfg  # SimpleNamespace or real config, both work here

        (output / subfolder).mkdir(parents=True, exist_ok=True)
        models_and_onnx_configs[name] = (mod, onnx_cfg)
        ordered_names.append(name)
        subfolder_map[name] = subfolder
        io_binding_outputs[name] = outputs

    # 6. Export each submodule to ONNX individually.
    # We export per-model (not via export_models) so we can pass return_dict=False
    # only for models whose forward signature actually has that parameter.
    # optimum's override_arguments injects model_kwargs unconditionally into
    # **kwargs even when the parameter is absent, causing TypeError.
    if models_and_onnx_configs:
        from optimum.exporters.onnx.convert import export as onnx_export_one
        opset = max(cfg.DEFAULT_ONNX_OPSET for _, cfg in models_and_onnx_configs.values())
        for n in ordered_names:
            mod_n, onnx_cfg_n = models_and_onnx_configs[n]
            # Use the wrapper module if available (e.g. vae_decoder: pqc + decoder)
            export_mod = _vae_decode_export_overrides.get(n, mod_n)
            # Ensure the export module has a fake config for export_pytorch
            if not hasattr(export_mod, "config"):
                export_mod.config = mod_n.config if hasattr(mod_n, "config") else SimpleNamespace()
            has_return_dict = "return_dict" in inspect.signature(export_mod.forward).parameters
            mk = {"return_dict": False} if has_return_dict else {}
            onnx_path = output / subfolder_map[n] / ONNX_WEIGHTS_NAME
            # Disable constant folding for text encoders — T5/UMT5 models have
            # relative position attention biases that get precomputed into very large
            # ONNX constants when constant folding is enabled, inflating the model size.
            is_text_enc = "text_encoder" in n
            onnx_export_one(
                model=export_mod,
                config=onnx_cfg_n,
                output=onnx_path,
                opset=opset,
                disable_dynamic_axes_fix=True,
                model_kwargs=mk,
                do_constant_folding=not is_text_enc,
            )

    # 7. Save per-submodule config.json (required by ORTModelMixin.__init__)
    for name, mod, subfolder, config_src in specs:
        if name not in models_and_onnx_configs:
            continue
        out_sub = output / subfolder
        if hasattr(config_src, "save_config"):
            config_src.save_config(out_sub)
        elif hasattr(config_src, "config") and hasattr(config_src.config, "save_pretrained"):
            config_src.config.save_pretrained(str(out_sub))
        elif hasattr(mod, "config") and hasattr(mod.config, "save_pretrained"):
            mod.config.save_pretrained(str(out_sub))

    # 8. Save pipeline-level components so ORTDiffusionPipeline can load them
    pt_pipeline.save_config(output)
    for attr in ("scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"):
        comp = getattr(pt_pipeline, attr, None)
        if comp is not None and hasattr(comp, "save_pretrained"):
            comp.save_pretrained(output / attr)

    # 9. Write io_binding output-shape files consumed by ORTSessionMixin
    io_dir = output / "io_binding"
    io_dir.mkdir(exist_ok=True)
    for name, out_shapes in io_binding_outputs.items():
        (io_dir / f"{name}_outputs.json").write_text(
            json.dumps({k: list(v) for k, v in out_shapes.items()}, indent=4)
        )


def _make_ort_pipeline_class(diffusers_class: type) -> type:
    """Dynamically create an ORT pipeline class for any diffusers pipeline.

    Returns a class that inherits from both ORTDiffusionPipeline and the given
    diffusers pipeline class, so it runs inference via ONNX Runtime while
    keeping the full diffusers pipeline API (schedulers, tokenizers, etc.).

    No manual subclass is needed when a new diffusers pipeline is released.
    """
    return type(
        f"ORT{diffusers_class.__name__}",
        (ORTDiffusionPipeline, diffusers_class),
        {"auto_model_class": diffusers_class, "task": "auto"},
    )


class ORTDiffusionPipeline(ORTParentMixin, DiffusionPipeline):
    """Generic ONNX Runtime pipeline for any diffusers DiffusionPipeline.

    Export any diffusers pipeline to ONNX and run it with ONNX Runtime — no
    model-specific subclass required.  A compatible class is created on the fly
    from the pipeline's ``_class_name`` config entry.

    Usage::

        pipe = ORTDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            export=True,
        )
        image = pipe("a photo of an astronaut").images[0]
    """

    config_name = DIFFUSION_PIPELINE_CONFIG_FILE_NAME
    task = "auto"
    library = "diffusers"
    auto_model_class = DiffusionPipeline

    def __init__(
        self,
        *,
        unet_session: InferenceSession | None = None,
        transformer_session: InferenceSession | None = None,
        vae_decoder_session: InferenceSession | None = None,
        vae_encoder_session: InferenceSession | None = None,
        text_encoder_session: InferenceSession | None = None,
        text_encoder_2_session: InferenceSession | None = None,
        text_encoder_3_session: InferenceSession | None = None,
        scheduler: SchedulerMixin | None = None,
        tokenizer: CLIPTokenizer | None = None,
        tokenizer_2: CLIPTokenizer | None = None,
        tokenizer_3: CLIPTokenizer | None = None,
        feature_extractor: CLIPFeatureExtractor | None = None,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: bool | None = None,
        use_io_binding: bool | None = None,
        model_save_dir: str | Path | None = None,
        **kwargs,
    ):
        self.unet = ORTUnet(unet_session, self, use_io_binding) if unet_session is not None else None
        self.transformer = ORTTransformer(transformer_session, self, use_io_binding) if transformer_session is not None else None
        self.text_encoder = ORTTextEncoder(text_encoder_session, self, use_io_binding) if text_encoder_session is not None else None
        self.text_encoder_2 = ORTTextEncoder(text_encoder_2_session, self, use_io_binding) if text_encoder_2_session is not None else None
        self.text_encoder_3 = ORTTextEncoder(text_encoder_3_session, self, use_io_binding) if text_encoder_3_session is not None else None
        self.vae_encoder = ORTVaeEncoder(vae_encoder_session, self, use_io_binding) if vae_encoder_session is not None else None
        self.vae_decoder = ORTVaeDecoder(vae_decoder_session, self, use_io_binding) if vae_decoder_session is not None else None

        super().initialize_ort_attributes(
            parts=list(filter(None, {
                self.unet, self.transformer,
                self.vae_encoder, self.vae_decoder,
                self.text_encoder, self.text_encoder_2, self.text_encoder_3,
            }))
        )

        self.vae = (
            ORTVae(self.vae_encoder, self.vae_decoder)
            if self.vae_encoder is not None or self.vae_decoder is not None
            else None
        )

        self.image_encoder = kwargs.pop("image_encoder", None)
        self.safety_checker = kwargs.pop("safety_checker", None)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.feature_extractor = feature_extractor

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "tokenizer_3": self.tokenizer_3,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }
        diffusers_pipeline_args = {
            k: v for k, v in all_pipeline_init_args.items()
            if k in inspect.signature(self.auto_model_class).parameters
        }
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        self.model_save_dir = model_save_dir

    @property
    def components(self) -> dict[str, Any]:
        components = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
        }
        return {k: v for k, v in components.items() if v is not None}

    def to(self, device: torch.device | str | int):
        for component in self.components.values():
            if isinstance(component, (ORTSessionMixin, ORTParentMixin)):
                component.to(device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        export: bool | None = None,
        export_by_inference: bool = False,
        inference_kwargs: dict[str, Any] | None = None,
        module_fixed_axis_fields: dict[str, list[str]] | None = None,
        skip_random_generation: bool = False,
        n_trials: int = 3,
        provider: str = "CPUExecutionProvider",
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        session_options: SessionOptions | None = None,
        use_io_binding: bool | None = None,
        **kwargs,
    ):
        # Inference-driven export always requires a fresh export pass
        if export_by_inference:
            export = True

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )

        hf_api = HfApi(user_agent=http_user_agent())
        hub_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download"),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir"),
            "revision": kwargs.get("revision"),
            "proxies": kwargs.get("proxies"),
            "token": kwargs.get("token"),
        }

        config = cls.load_config(model_name_or_path, **hub_kwargs)
        config = config[0] if isinstance(config, tuple) else config

        model_save_path = Path(model_name_or_path)

        # auto-detect whether export is needed
        if export is None:
            if "unet" in config and config["unet"] is not None:
                relative_file_path = Path(DIFFUSION_MODEL_UNET_SUBFOLDER) / ONNX_WEIGHTS_NAME
            elif "transformer" in config and config["transformer"] is not None:
                relative_file_path = Path(DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER) / ONNX_WEIGHTS_NAME
            else:
                raise ValueError(
                    "Neither 'unet' nor 'transformer' found in pipeline config. "
                    "Set export=True or export=False explicitly."
                )
            absolute_file_path = model_save_path / relative_file_path
            export = not (
                absolute_file_path.is_file()
                or hf_api.file_exists(
                    repo_id=str(model_name_or_path),
                    filename=str(relative_file_path),
                    revision=hub_kwargs.get("revision"),
                    token=hub_kwargs.get("token"),
                )
            )

        if export:
            model_save_path = Path("/dev/shm")

            torch_dtype = kwargs.pop("torch_dtype", None)

            if export_by_inference:
                _on_the_fly_diffusion_export(
                    model_name_or_path=model_name_or_path,
                    output=model_save_path,
                    inference_kwargs=inference_kwargs or {},
                    module_fixed_axis_fields=module_fixed_axis_fields,
                    torch_dtype=torch_dtype,
                    device=get_device_for_provider(provider, {}).type,
                    hub_kwargs=hub_kwargs,
                    n_trials=n_trials,
                    skip_random_generation=skip_random_generation,
                )
            else:
                if torch_dtype is not None:
                    if torch_dtype == torch.float16:
                        kwargs["dtype"] = "fp16"
                    elif torch_dtype == torch.float32:
                        kwargs["dtype"] = "fp32"
                    else:
                        raise ValueError(f"Unsupported torch_dtype for export: {torch_dtype}")

                export_kwargs = {
                    "slim": kwargs.pop("slim", False),
                    "dtype": kwargs.pop("dtype", None),
                    "device": get_device_for_provider(provider, {}).type,
                    "no_dynamic_axes": kwargs.pop("no_dynamic_axes", False),
                }
                main_export(
                    model_name_or_path=str(model_name_or_path),
                    output=model_save_path,
                    no_post_process=True,
                    do_validation=False,
                    task="auto",
                    **{k: v for k, v in export_kwargs.items() if v is not None},
                    **hub_kwargs,
                )

        # download model from hub if it's not a local directory
        if not model_save_path.is_dir():
            all_components = {key for key in config if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update({
                ONNX_WEIGHTS_NAME, DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
                SCHEDULER_CONFIG_NAME, CONFIG_NAME,
            })
            model_save_folder = hf_api.snapshot_download(
                repo_id=str(model_name_or_path),
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
                **hub_kwargs,
            )
            model_save_path = Path(model_save_folder)

        model_paths = {
            "unet":          model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "transformer":   model_save_path / DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "vae_encoder":   model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "vae_decoder":   model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder":  model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder_2": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder_3": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER / ONNX_WEIGHTS_NAME,
        }

        sessions = {}
        models = {}
        for model_key, path in model_paths.items():
            if kwargs.get(model_key) is not None:
                models[model_key] = kwargs.pop(model_key)
            elif kwargs.get(f"{model_key}_session") is not None:
                sessions[f"{model_key}_session"] = kwargs.pop(f"{model_key}_session")
            elif path.is_file():
                sessions[f"{model_key}_session"] = InferenceSession(
                    path,
                    providers=providers,
                    provider_options=provider_options,
                    sess_options=session_options,
                )

        submodels = {}
        for submodel in {"scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"}:
            if kwargs.get(submodel) is not None:
                submodels[submodel] = kwargs.pop(submodel)
            elif config.get(submodel, (None, None))[0] is not None:
                library_name, library_classes = config.get(submodel)
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                if (model_save_path / submodel).is_dir():
                    submodels[submodel] = class_obj.from_pretrained(model_save_path / submodel)
                else:
                    submodels[submodel] = class_obj.from_pretrained(model_save_path)

        # Resolve the concrete pipeline class dynamically — no hardcoded mapping needed.
        # For any new diffusers pipeline, the right ORT class is created on the fly.
        if cls is ORTDiffusionPipeline:
            pipeline_class_name = config["_class_name"]
            diffusers_class = getattr(diffusers, pipeline_class_name, None)
            if diffusers_class is None:
                raise ValueError(
                    f"Pipeline class '{pipeline_class_name}' not found in diffusers. "
                    f"Make sure diffusers is up to date."
                )
            ort_pipeline_class = _make_ort_pipeline_class(diffusers_class)
        else:
            ort_pipeline_class = cls

        ort_pipeline = ort_pipeline_class(
            **sessions,
            **submodels,
            **models,
            use_io_binding=use_io_binding,
            **kwargs,
        )

        ort_pipeline.register_to_config(**config)
        ort_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", str(model_name_or_path)))

        # Wire up IO binding shape files for each component
        io_binding_dir = model_save_path / "io_binding"
        for key, comp in ort_pipeline.components.items():
            if key == "vae":
                if comp.encoder is not None:
                    comp.encoder.set_io_binding_file(str(io_binding_dir / "vae_encoder_outputs.json"))
                if comp.decoder is not None:
                    comp.decoder.set_io_binding_file(str(io_binding_dir / "vae_decoder_outputs.json"))
            else:
                comp.set_io_binding_file(str(io_binding_dir / f"{key}_outputs.json"))

        return ort_pipeline

    def save_pretrained(self, save_directory: str | Path, push_to_hub: bool = False, **kwargs):
        model_save_path = Path(save_directory)
        model_save_path.mkdir(parents=True, exist_ok=True)

        self.save_config(model_save_path)
        self.scheduler.save_pretrained(model_save_path / "scheduler")

        for attr, subfolder in [
            ("unet",          DIFFUSION_MODEL_UNET_SUBFOLDER),
            ("transformer",   DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER),
            ("vae_encoder",   DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER),
            ("vae_decoder",   DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER),
            ("text_encoder",  DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER),
            ("text_encoder_2", DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER),
            ("text_encoder_3", DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER),
        ]:
            component = getattr(self, attr, None)
            if component is not None:
                component.save_pretrained(model_save_path / subfolder)

        for attr in ("image_encoder", "safety_checker", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"):
            component = getattr(self, attr, None)
            if component is not None:
                component.save_pretrained(model_save_path / attr)
