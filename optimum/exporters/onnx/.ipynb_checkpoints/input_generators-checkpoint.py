from __future__ import annotations

import torch
from optimum.utils.input_generators import DummyInputGenerator

class DummyTupleInputGenerator(DummyInputGenerator):
    """Generates dummy tensors from traced (shape, dtype) tuples."""

    SUPPORTED_INPUT_NAMES = (".*",)

    def __init__(self, task: str, config_dim: dict[str, int] | None = None, **kwargs):
        super().__init__()
        self.config_dim = config_dim or {}

    def generate(
        self,
        input_name: str,
        tensor_shape: tuple[int, ...],
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ) -> torch.Tensor:
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
