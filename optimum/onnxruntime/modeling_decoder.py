from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from optimum.onnxruntime import ORTModelForCausalLM

from .modeling import _OnTheFlyORTMixin

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class OnTheFlyORTModelForCausalLM(_OnTheFlyORTMixin, ORTModelForCausalLM):
    """ORTModelForCausalLM with inference-driven ONNX export.

    Traces actual tensor shapes (including position_ids and past_key_values)
    via a two-step prefill + decode forward, then exports to ONNX and loads
    with ORT — no changes to optimum-onnx required.
    """
