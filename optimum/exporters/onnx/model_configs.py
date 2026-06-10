from __future__ import annotations

import importlib.util
import os
import sys

# Re-export everything from the real site-packages model_configs so that
# code that imports from 'optimum.exporters.onnx.model_configs' (like the real
# package's convert.py) still finds SpeechT5OnnxConfig, BertOnnxConfig, etc.
def _load_real_model_configs():
    _key = "_idmc_real_onnx_model_configs"
    if _key in sys.modules:
        return sys.modules[_key]
    for _p in sys.path:
        if not _p or "inference_driven_model_compiler" in _p:
            continue
        _f = os.path.join(_p, "optimum", "exporters", "onnx", "model_configs.py")
        if os.path.exists(_f):
            spec = importlib.util.spec_from_file_location(_key, _f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_key] = mod
            spec.loader.exec_module(mod)
            return mod
    return None

_real = _load_real_model_configs()
if _real is not None:
    import types
    _g = globals()
    for _name, _val in vars(_real).items():
        if not _name.startswith("_"):
            _g[_name] = _val

import re
from typing import Any

import torch
from optimum.exporters.onnx.base import OnnxConfig
from optimum.utils.normalized_config import NormalizedConfig

from .input_generators import DummyTupleInputGenerator


class _AnyNormalizedConfig(NormalizedConfig):
    """Pass-through normalized config that works for any model type."""


def _get_compatible_model_patcher():
    """Return a ModelPatcher that skips SDPA mask registration on transformers>=5.x.

    Optimum's stock ModelPatcher registers sdpa_mask_without_vmap as the SDPA mask
    function. That function requires cache_position as a positional arg, but newer
    transformers (>=5.0) no longer passes cache_position to the mask interface —
    it passes q_length/q_offset instead. Skipping the registration lets the stock
    sdpa_mask handle tracing correctly.
    """
    from optimum.exporters.onnx.model_patcher import ModelPatcher
    from optimum.utils import is_transformers_version

    if is_transformers_version("<", "5.0"):
        return ModelPatcher

    class _CompatModelPatcher(ModelPatcher):
        def __enter__(self):
            result = super().__enter__()
            # The parent (ModelPatcher) registered sdpa_mask_without_vmap and
            # eager_mask_without_vmap, which require cache_position as a positional
            # arg. Transformers>=5 passes q_length/q_offset instead, so we restore
            # the compatible originals.
            try:
                from transformers.masking_utils import (
                    ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask, eager_mask,
                )
                ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = sdpa_mask
                ALL_MASK_ATTENTION_FUNCTIONS["eager"] = eager_mask
            except Exception:
                pass
            return result

    return _CompatModelPatcher


class DummyOnnxConfig(OnnxConfig):
    """ONNX config built entirely from traced inference shapes.

    Works with any encoder or decoder model without requiring a hand-written
    model-specific OnnxConfig subclass.
    """

    NORMALIZED_CONFIG_CLASS = _AnyNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTupleInputGenerator,)
    _MODEL_PATCHER = _get_compatible_model_patcher()

    def __init__(
        self,
        config,
        task: str = "backbone",
        preprocessors=None,
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        model_inputs: dict[str, Any] | None = None,
        model_outputs: dict[str, Any] | None = None,
        config_dim: dict[str, int] | None = None,
        dynamic_axes: "dict[str, dict[int, str]] | None" = None,
        model_input_dtypes: "dict[str, torch.dtype] | None" = None,
    ):
        super().__init__(
            config=config,
            task=task,
            preprocessors=preprocessors,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
        )
        self.task = task
        self.model_inputs    = model_inputs  or {}
        self.model_outputs   = model_outputs or {}
        self.config_dim      = config_dim    or {}
        self._dynamic_axes   = dynamic_axes  or {}   # empirically derived
        self._input_gen      = DummyTupleInputGenerator(
            task=task, config_dim=self.config_dim, input_dtypes=model_input_dtypes
        )

        # Tell ModelPatcher to enable KV-cache output when past_key_values are present
        has_kv = any(k.startswith("past_key_values.") for k in self.model_inputs)
        self.use_past           = has_kv
        self.use_past_in_inputs = has_kv

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_kv(name: str) -> bool:
        return bool(re.match(r"^(past_key_values|present)\.\d+\.(key|value)$", name))

    def flatten_output_collection_property(self, name: str, field):
        """Flatten KV-cache collections using optimum's ``.key``/``.value`` naming.

        The base ``OnnxConfig`` flattens a tuple-of-tuples generically into
        ``{name}.0``, ``{name}.1``, … which does not match the exported graph's
        ``past_key_values.{i}.key`` / ``present.{i}.value`` input/output names.
        During validation optimum flattens both the reference inputs
        (``past_key_values``) and the reference outputs (``present``) with this
        method, so emitting the per-layer key/value names here makes the feed and
        the output-name comparison line up with the actual ONNX graph.
        """
        if name in ("present", "past_key_values"):
            flattened = {}
            for idx, t in enumerate(field):
                flattened[f"{name}.{idx}.key"] = t[0]
                flattened[f"{name}.{idx}.value"] = t[1]
            return flattened
        return super().flatten_output_collection_property(name, field)

    def _axes_for(self, name: str, shape: tuple) -> dict[int, str]:
        """Return dynamic-axes dict for one tensor.

        Priority:
        1. Empirically derived axes from multi-trial tracing (``_dynamic_axes``).
        2. Fallback heuristic: dim-0 always dynamic; other dims dynamic only if
           they don't match any fixed config dimension value.
        """
        shape = tuple(shape)
        ndim = len(shape)

        if name in self._dynamic_axes:
            # Drop any axis index beyond this tensor's rank — e.g. a 0-d scalar
            # diffusion timestep that would otherwise carry a batch axis, which
            # torch.onnx rejects ("Dynamic shape axis should be no more than the
            # shape dimension").
            return {k: v for k, v in self._dynamic_axes[name].items() if k < ndim}

        # Scalars (0-d) have no axis to mark dynamic.
        if ndim == 0:
            return {}

        # ── fallback heuristic (single-trial or missing) ──
        axes: dict[int, str] = {0: "batch"}
        for idx, dim in enumerate(shape):
            if idx == 0:
                continue
            if self._is_kv(name) and idx == 2:
                axes[idx] = f"{name}_dim_{idx}"
                continue
            if not any(v == dim for v in self.config_dim.values()):
                axes[idx] = f"{name}_dim_{idx}"
        return axes

    # ── OnnxConfig interface ───────────────────────────────────────────────

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        result = {}
        for name, val in self.model_inputs.items():
            shape = tuple(val.shape) if isinstance(val, torch.Tensor) else val
            result[name] = self._axes_for(name, shape)
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {name: self._axes_for(name, shape) for name, shape in self.model_outputs.items()}

    def generate_dummy_inputs(
        self,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        **kwargs,
    ) -> dict[str, Any]:
        # Prefer the dtype stored at construction time (detected from the actual model)
        # over the caller's default "fp32" so fp16 models export with fp16 dummy inputs.
        float_dtype = self.float_dtype

        kv: dict[tuple, torch.Tensor] = {}
        flat: dict[str, torch.Tensor] = {}

        for name, val in self.model_inputs.items():
            shape = tuple(val.shape) if isinstance(val, torch.Tensor) else val
            tensor = self._input_gen.generate(name, shape, framework=framework,
                                              int_dtype=int_dtype, float_dtype=float_dtype)
            m = re.match(r"^past_key_values\.(\d+)\.(key|value)$", name)
            if m:
                kv[(int(m.group(1)), m.group(2))] = tensor
            else:
                flat[name] = tensor

        if kv:
            n_layers = max(i for i, _ in kv) + 1
            flat["past_key_values"] = tuple(
                (kv[(i, "key")], kv[(i, "value")]) for i in range(n_layers)
            )

        return flat
