from __future__ import annotations

import importlib.util
import os
import sys

# Re-export everything from the real site-packages input_generators so that code
# that imports DummyMoonshineAudioInputGenerator etc. still works.
def _load_real_input_generators():
    _key = "_idmc_real_onnx_input_generators"
    if _key in sys.modules:
        return sys.modules[_key]
    for _p in sys.path:
        if not _p or "inference_driven_model_compiler" in _p:
            continue
        _f = os.path.join(_p, "optimum", "exporters", "onnx", "input_generators.py")
        if os.path.exists(_f):
            spec = importlib.util.spec_from_file_location(_key, _f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_key] = mod
            spec.loader.exec_module(mod)
            return mod
    return None

_real_ig = _load_real_input_generators()
if _real_ig is not None:
    _g = globals()
    for _name, _val in vars(_real_ig).items():
        if not _name.startswith("_"):
            _g[_name] = _val

import torch
from optimum.utils.input_generators import DummyInputGenerator

class DummyTupleInputGenerator(DummyInputGenerator):
    """Generates dummy tensors from traced (shape, dtype) tuples."""

    SUPPORTED_INPUT_NAMES = (".*",)

    def __init__(
        self,
        task: str,
        config_dim: dict[str, int] | None = None,
        input_dtypes: dict[str, "torch.dtype"] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config_dim = config_dim or {}
        # Per-input exact dtypes captured from the real traced tensors. When an
        # input is present here its dtype is authoritative: it decides int-vs-float
        # generation and the final tensor is cast to the exact captured dtype.
        # This is essential for diffusion submodules where the name-based heuristic
        # would misclassify (e.g. SDXL's float "time_ids" matches the "_id" rule).
        self.input_dtypes = input_dtypes or {}

    def generate(
        self,
        input_name: str,
        tensor_shape: tuple[int, ...],
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
        # When the captured dtype is known, it overrides the name heuristic.
        captured_dtype = self.input_dtypes.get(input_name)
        if captured_dtype is not None:
            if captured_dtype.is_floating_point:
                tensor = self.random_float_tensor(
                    list(tensor_shape), framework=framework, dtype=float_dtype
                )
            else:
                max_val = (
                    max(self.config_dim.get("vocab_size", 1000), 1)
                    if "input_id" in input_name
                    else max(max(tensor_shape, default=1), 1)
                )
                tensor = self.random_int_tensor(
                    list(tensor_shape),
                    max_value=max_val,
                    min_value=0,
                    framework=framework,
                    dtype=int_dtype,
                )
            return tensor.to(captured_dtype) if framework == "pt" else tensor

        # integer tensors: any *_id(s)*, masks, positions
        if "_id" in input_name or "mask" in input_name or "position" in input_name:
            if "input_id" in input_name:
                max_val = max(self.config_dim.get("vocab_size", 1000), 1)
            elif "token_type" in input_name:
                max_val = 1   # BERT-style: 0 or 1
            else:
                max_val = max(max(tensor_shape), 1)
            return self.random_int_tensor(
                list(tensor_shape),
                max_value=max_val,
                min_value=0,
                framework=framework,
                dtype=int_dtype,
            )
        return self.random_float_tensor(list(tensor_shape), framework=framework, dtype=float_dtype)
