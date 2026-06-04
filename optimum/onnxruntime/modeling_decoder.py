from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.utils.save_utils import maybe_save_preprocessors

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# The inference-driven exporter creates "transformer.onnx" as the module key.
_INFERENCE_EXPORT_FILE = "transformer.onnx"


class OnTheFlyORTModelForCausalLM(ORTModelForCausalLM):
    """ORTModelForCausalLM with inference-driven ONNX export.

    Accepts inference_kwargs and module_fixed_axis_fields to trace actual
    tensor shapes at inference time and use them for ONNX export.
    Because the export traces a plain forward pass (no past-KV), use_cache
    defaults to False.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | Path,
        config: "PretrainedConfig | None" = None,
        export: bool = False,
        inference_kwargs: "dict[str, Any] | None" = None,
        module_fixed_axis_fields: "dict[str, list[str]] | None" = None,
        export_by_inference: bool = False,
        skip_random_generation: bool = False,
        # inference-driven export traces a plain forward (no past-KV)
        use_cache: bool = False,
        **kwargs,
    ):
        return super().from_pretrained(
            model_id,
            config=config,
            export=export,
            inf_kwargs=inference_kwargs,
            module_arch_fields=module_fixed_axis_fields,
            export_by_inference=export_by_inference,
            skip_random_generation=skip_random_generation,
            use_cache=use_cache,
            **kwargs,
        )

    @classmethod
    def _export(
        cls,
        model_id: str | Path,
        config: "PretrainedConfig",
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = "",
        token: "bool | str | None" = None,
        use_cache: bool = False,
        inf_kwargs: "dict[str, Any] | None" = None,
        module_arch_fields: "dict[str, Any] | None" = None,
        export_by_inference: bool = False,
        skip_random_generation: bool = False,
        **kwargs,
    ) -> "OnTheFlyORTModelForCausalLM":
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        cache_dir = cache_dir or HUGGINGFACE_HUB_CACHE

        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
        # inference-driven export produces a plain forward; keep task without -with-past
        # so the ORT model loads without expecting KV inputs

        save_dir_path = Path("/dev/shm")

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=True,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            inf_kwargs=inf_kwargs,
            module_arch_fields=module_arch_fields,
            export_by_inference=export_by_inference,
            skip_random_generation=skip_random_generation,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            file_name=_INFERENCE_EXPORT_FILE,
            model_save_dir=save_dir_path,
            **kwargs,
        )
