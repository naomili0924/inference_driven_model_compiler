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


class _InFeaturesShim:
    """Minimal stand-in exposing ``add_embedding.linear_1.in_features``.

    SDXL's ``_get_add_time_ids`` validates the micro-conditioning embedding size
    against ``self.unet.add_embedding.linear_1.in_features``.  The real Linear
    layer lives inside the (now ONNX) UNet, so we surface the same integer from
    the saved config (``projection_class_embeddings_input_dim``).
    """

    def __init__(self, in_features: int):
        self.linear_1 = type("_Linear", (), {"in_features": in_features})()


class ORTUnet(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "time_cond_proj_dim"):
            self.register_to_config(time_cond_proj_dim=None)

        # SDXL-style UNets carry an "add_embedding" used for micro-conditioning.
        # The pipeline only inspects its input feature count, which equals the
        # config's projection_class_embeddings_input_dim.
        add_embed_dim = getattr(self.config, "projection_class_embeddings_input_dim", None)
        if add_embed_dim is not None:
            self.add_embedding = _InFeaturesShim(add_embed_dim)

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
        # SDXL marks its VAE force_upcast=True so diffusers re-runs the decoder in
        # fp32. That path dereferences self.vae.post_quant_conv.parameters(), which
        # the ONNX VAE doesn't expose, and the exported graph already runs at its
        # native (fp16) precision. Disable upcasting so the pipeline feeds the ONNX
        # decoder directly.
        if getattr(self.config, "force_upcast", False):
            self.register_to_config(force_upcast=False)
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


class _UnetAddedCondWrapper(torch.nn.Module):
    """Export wrapper for UNet denoisers that take ``added_cond_kwargs``.

    SDXL-style UNets receive their pooled text embedding and micro-conditioning
    (``text_embeds`` + ``time_ids``) inside a nested ``added_cond_kwargs`` dict.
    The inference tracer only forwards *tensor* arguments, so the dict is dropped
    and the bare UNet call would fail.  This wrapper exposes those tensors as
    first-class arguments and rebuilds the dict internally, so they become
    ordinary ONNX inputs named ``text_embeds`` / ``time_ids`` — exactly the names
    ``ORTUnet.forward`` already feeds back in via ``added_cond_kwargs``.

    The single decoded tensor is returned under the key ``out_sample`` so the
    exported graph's output name matches what ``ORTUnet.forward`` expects.
    """

    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds=None,
        time_ids=None,
        timestep_cond=None,
    ):
        added_cond_kwargs = {}
        if text_embeds is not None:
            added_cond_kwargs["text_embeds"] = text_embeds
        if time_ids is not None:
            added_cond_kwargs["time_ids"] = time_ids
        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs or None,
            return_dict=False,
        )
        sample_out = out[0] if isinstance(out, (list, tuple)) else out
        return {"out_sample": sample_out}


class _TextEncoderHiddenStatesWrapper(torch.nn.Module):
    """Export wrapper that forces ``output_hidden_states=True`` on a text encoder.

    SDXL builds its prompt embedding from the *penultimate* hidden layer of each
    CLIP text encoder and its pooled embedding from the projection output, so the
    exported graph must expose every hidden state plus the pooled output — none of
    which the encoder emits with its default arguments.  The captured outputs are
    flattened into ``hidden_states.{i}`` entries (the layout ``ORTTextEncoder``
    reconstructs), and the model's own output field order is preserved so that
    ``output[0]`` keeps its native meaning (``text_embeds`` for a projection head,
    ``last_hidden_state`` otherwise).
    """

    def __init__(self, text_encoder: torch.nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids, attention_mask=None):
        out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        result = {}
        for key, value in out.items():
            if key == "hidden_states" and isinstance(value, (list, tuple)):
                for i, hs in enumerate(value):
                    result[f"hidden_states.{i}"] = hs
            elif torch.is_tensor(value):
                result[key] = value
        return result


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
    _export_overrides: dict[str, torch.nn.Module] = {}
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
            _export_overrides = {"vae_decoder": _vae_full_decode}

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
            _export_overrides = {"vae_decoder": _vae_full_decode}
        else:
            _export_overrides = {}

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

    # Keep the captured submodules and tensors on the export device.  Tracing and
    # torch.onnx.export both run real forward passes, and on CPU fp16 (which the
    # SDXL VAE decoder and large UNets use) those passes are pathologically slow —
    # an fp16 export that finishes in seconds on CUDA can take tens of minutes on
    # CPU.  optimum's export() moves the (CPU-generated) dummy inputs to this same
    # device, so GPU export is consistent.  Falls back to CPU when device=="cpu".
    export_device = torch.device(device)

    def _to_dev(v):
        if torch.is_tensor(v):
            return v.to(export_device)
        if isinstance(v, list):
            return [_to_dev(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_to_dev(x) for x in v)
        if isinstance(v, dict):
            return {k2: _to_dev(v2) for k2, v2 in v.items()}
        return v

    for name, mod, _, _ in specs:
        if name in captured:
            mod.to(export_device)
            captured[name] = {k: _to_dev(v) for k, v in captured[name].items()}
    # Keep the full vae_decoder wrapper (pqc + decoder) on the export device too.
    for _, wrapper_mod in _export_overrides.items():
        wrapper_mod.to(export_device)

    # Normalize the vae_decoder captured input key to "z" when exported via
    # _VaeNoQuantDecodeWrapper (CogVideoX-style, no post_quant_conv).  The hook
    # names the key after the decoder's first parameter ("sample" for CogVideoX),
    # but the wrapper's forward signature uses "z" so trace_model_shapes must
    # receive {"z": tensor}.
    if "vae_decoder" in captured and "vae_decoder" in _export_overrides:
        wrapper = _export_overrides["vae_decoder"]
        # The capture hook names the input after the *hooked* module's forward
        # parameter — e.g. "input" for an nn.Conv2d post_quant_conv (SDXL/SD), or
        # "sample" for a CogVideoX decoder.  The export wrapper's own forward uses
        # a different name ("x" for _VaeFullDecodeWrapper, "z" for the no-quant
        # variant), so rename the captured tensor key to the wrapper's first
        # parameter — otherwise trace_model_shapes' keyword call raises.
        wrapper_params = list(inspect.signature(wrapper.forward).parameters)
        target_key = wrapper_params[0] if wrapper_params else None
        cap = captured["vae_decoder"]
        tensor_keys = [k for k, v in cap.items() if torch.is_tensor(v)]
        if target_key and tensor_keys and tensor_keys[0] != target_key:
            cap[target_key] = cap.pop(tensor_keys[0])
            captured["vae_decoder"] = cap

    # 4b. Register export wrappers for UNet-based (SDXL-style) pipelines.
    #
    #  • UNet: lift the nested ``added_cond_kwargs`` dict (text_embeds + time_ids)
    #    into top-level captured tensors so they become first-class ONNX inputs,
    #    and export through _UnetAddedCondWrapper which rebuilds the dict.
    #  • Text encoders: export through _TextEncoderHiddenStatesWrapper so the graph
    #    exposes every hidden state + pooled output (SDXL uses the penultimate
    #    hidden layer and the pooled projection).
    _modules_by_name = {name: mod for name, mod, _, _ in specs}

    for unet_name in ("unet",):
        cap = captured.get(unet_name)
        if cap is None or unet_name in _export_overrides:
            continue
        added = cap.pop("added_cond_kwargs", None)
        if isinstance(added, dict):
            for k, v in added.items():
                if torch.is_tensor(v):
                    cap[k] = v
        # Only wrap when the UNet actually consumes added_cond_kwargs; otherwise
        # the plain UNet export (SD1.x-style) already works.
        if isinstance(added, dict) and any(torch.is_tensor(v) for v in added.values()):
            _export_overrides[unet_name] = _UnetAddedCondWrapper(
                _modules_by_name[unet_name]
            ).to(export_device)

    for te_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        cap = captured.get(te_name)
        if cap is None or te_name in _export_overrides:
            continue
        # Wrap only when the pipeline asked for hidden states (SDXL); plain text
        # encoders that only need last_hidden_state (e.g. WAN/UMT5) are exported
        # as-is so their existing behaviour is unchanged.
        if cap.get("output_hidden_states"):
            _export_overrides[te_name] = _TextEncoderHiddenStatesWrapper(
                _modules_by_name[te_name]
            ).to(export_device)

    # 4c. Align every captured float tensor with its module's parameter dtype.
    # The pipeline may feed a submodule inputs in a different precision than the
    # module's own weights — most notably SDXL, which upcasts its VAE to fp32 for
    # decoding while the exported module stays fp16.  Tracing/exporting with the
    # mismatched dtype crashes ("Input type (float) and bias type (Half) should be
    # the same"), so cast captured floats to the module's dtype here.  This runs
    # after the UNet's added_cond tensors have been lifted into `captured`.
    for name, mod, _, config_src in specs:
        if name not in captured:
            continue
        target_module = _export_overrides.get(name, mod)
        # SDXL-family VAEs are numerically unstable in fp16 — that is exactly why the
        # diffusers pipeline sets force_upcast and decodes the VAE in fp32.  Export
        # the VAE decoder in fp32 so the ONNX graph reproduces the upcast reference
        # instead of emitting NaNs; the fp16 latents are auto-cast to fp32 at
        # inference by the ORT session's input handling.
        force_upcast = bool(
            getattr(getattr(config_src, "config", None), "force_upcast", False)
        )
        if name == "vae_decoder" and force_upcast:
            target_module.to(torch.float32)
            mod_dtype = torch.float32
        else:
            mod_dtype = next(
                (p.dtype for p in target_module.parameters() if p.is_floating_point()),
                None,
            )
        if mod_dtype is None:
            continue
        captured[name] = {
            k: (v.to(mod_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in captured[name].items()
        }

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
        trace_mod = _export_overrides.get(name, mod)
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

        # For a *video* vae_decoder the captured input is a single chunk (e.g. 3
        # frames), but at inference the full temporal extent varies.  Force dim-2
        # (frames) to be dynamic so the exported ONNX accepts any number of frames.
        # Image VAEs (4-D latents B,C,H,W) have no temporal axis — skip them.
        if name == "vae_decoder":
            for inp_name, shape in inputs.items():
                shape_t = tuple(shape.shape) if isinstance(shape, torch.Tensor) else shape
                if len(shape_t) == 5:
                    dynamic_axes.setdefault(inp_name, {0: "batch"})[2] = "num_frames"

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

        # Map each input to the exact dtype of the captured (real) tensor so the
        # dummy inputs used for export carry the correct dtype.  Without this the
        # name heuristic would, e.g., generate SDXL's float "time_ids" as int64.
        model_input_dtypes = {
            k: v.dtype for k, v in captured[name].items() if torch.is_tensor(v)
        }

        onnx_cfg = DummyOnnxConfig(
            config=cfg,
            task="backbone",
            model_inputs=inputs,
            model_outputs=outputs,
            config_dim=config_dim,
            dynamic_axes=dynamic_axes,
            float_dtype=float_dtype,
            model_input_dtypes=model_input_dtypes,
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
            export_mod = _export_overrides.get(n, mod_n)
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
                device=str(export_device),
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
