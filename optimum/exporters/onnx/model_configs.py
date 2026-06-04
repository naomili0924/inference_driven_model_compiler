from __future__ import annotations

import re
from typing import Any

import torch
from optimum.exporters.onnx.base import OnnxConfig
from optimum.utils.normalized_config import NormalizedConfig

from .input_generators import DummyTupleInputGenerator


class _AnyNormalizedConfig(NormalizedConfig):
    """Pass-through normalized config that works for any model type."""


class DummyOnnxConfig(OnnxConfig):
    """ONNX config built entirely from traced inference shapes.

    Works with any encoder or decoder model without requiring a hand-written
    model-specific OnnxConfig subclass.
    """

    NORMALIZED_CONFIG_CLASS = _AnyNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTupleInputGenerator,)

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
    ):
        super().__init__(
            config=config,
            task=task,
            preprocessors=preprocessors,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
        )
        self.task = task
        self.model_inputs  = model_inputs  or {}
        self.model_outputs = model_outputs or {}
        self.config_dim    = config_dim    or {}
        self._input_gen    = DummyTupleInputGenerator(task=task, config_dim=self.config_dim)

        # Tell ModelPatcher to enable KV-cache output when past_key_values are present
        has_kv = any(k.startswith("past_key_values.") for k in self.model_inputs)
        self.use_past          = has_kv
        self.use_past_in_inputs = has_kv

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_kv(name: str) -> bool:
        return bool(re.match(r"^past_key_values\.\d+\.(key|value)$", name))

    def _infer_axes(self, shape: tuple[int, ...], name: str) -> dict[int, str]:
        """Mark batch (dim-0) and any non-fixed dim as dynamic."""
        axes: dict[int, str] = {}
        for idx, dim in enumerate(shape):
            if idx == 0:
                axes[0] = "batch"
                continue
            if self._is_kv(name) and idx == 2:
                # past-sequence length is always dynamic for KV tensors
                axes[idx] = f"{name}_dim_{idx}"
                continue
            # mark dynamic if it doesn't match any fixed config dim
            if not any(v == dim for v in self.config_dim.values()):
                axes[idx] = f"{name}_dim_{idx}"
        return axes

    # ── OnnxConfig interface ───────────────────────────────────────────────

    @property
    def inputs(self) -> dict[str, dict[int, str]]:
        result = {}
        for name, val in self.model_inputs.items():
            shape = tuple(val.shape) if isinstance(val, torch.Tensor) else val
            result[name] = self._infer_axes(shape, name)
        return result

    @property
    def outputs(self) -> dict[str, dict[int, str]]:
        return {name: self._infer_axes(shape, name) for name, shape in self.model_outputs.items()}

    def generate_dummy_inputs(
        self,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
        **kwargs,
    ) -> dict[str, Any]:
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
