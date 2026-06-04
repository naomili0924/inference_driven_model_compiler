from typing import Any
from pathlib import Path

from optimum.exporters.onnx import main_export as _upstream_main_export


def main_export(
    model_name_or_path: str,
    output: str | Path,
    task: str = "auto",
    inference_kwargs: "dict[str, Any] | None" = None,
    module_fixed_axis_fields: "dict[str, list[str]] | None" = None,
    export_by_inference: bool = False,
    skip_random_generation: bool = False,
    **kwargs,
):
    """Wrapper around optimum main_export with renamed parameters.

    Uses inference_kwargs / module_fixed_axis_fields instead of
    inf_kwargs / module_arch_fields.
    """
    return _upstream_main_export(
        model_name_or_path,
        output,
        task=task,
        inf_kwargs=inference_kwargs,
        module_arch_fields=module_fixed_axis_fields,
        export_by_inference=export_by_inference,
        skip_random_generation=skip_random_generation,
        **kwargs,
    )


__all__ = ["main_export"]
