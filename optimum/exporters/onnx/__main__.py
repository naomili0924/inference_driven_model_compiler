from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
if TYPE_CHECKING:
    from optimum.exporters.onnx.base import OnnxConfig

from optimum.utils.import_utils import (
    is_diffusers_available,
    is_sentence_transformers_available,
    is_timm_available,
    is_transformers_version,
)

import torch
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, AutoTokenizer

try:
    from transformers import Mxfp4Config
except ImportError:
    Mxfp4Config = None

from optimum.exporters.onnx.constants import SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED

from optimum.utils import logging
logger = logging.get_logger()

from optimum.utils.save_utils import maybe_load_preprocessors
from optimum.exporters.tasks import TasksManager

from optimum.exporters.utils import DisableCompileContextManager
from optimum.exporters.onnx.convert import onnx_export_from_model

def main_export(
    model_name_or_path: str,
    output: str | Path,
    task: str = "auto",
    opset: int | None = None,
    device: str = "cpu",
    dtype: str | None = None,
    optimize: str | None = None,
    monolith: bool = False,
    no_post_process: bool = False,
    framework: str | None = "pt",
    atol: float | None = None,
    pad_token_id: int | None = None,
    # inference kwargs
    inference_kwargs: dict[str, Any] | None = None,
    # module_arch_configs
    module_fixed_axis_fields: dict[str, list[str]] | None = None,
    # flag for export_by_inference
    export_by_inference: bool = False,
    skip_random_generation: bool = False,
    # names of inputs whose traced VALUES must be replayed verbatim (not randomized)
    fixed_inputs: list[str] | None = None,
    # hub options
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    token: bool | str | None = None,
    ########################################
    do_validation: bool = True,
    model_kwargs: dict[str, Any] | None = None,
    custom_onnx_configs: dict[str, OnnxConfig] | None = None,
    fn_get_submodels: Callable | None = None,
    use_subprocess: bool = False,
    _variant: str = "default",
    library_name: str | None = None,
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    slim: bool = False,
    dynamo: bool = False,
    **kwargs_shapes,
):
    if dtype is None:
        dtype = "fp32"  # Defaults to float32

    if optimize == "O4" and device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if library_name == "sentence_transformers" and not is_sentence_transformers_available():
        raise ImportError(
            "The library `sentence_transformers` was specified, but it is not installed. "
            "Please install it with `pip install sentence-transformers`."
        )

    if library_name == "diffusers" and not is_diffusers_available():
        raise ImportError(
            "The library `diffusers` was specified, but it is not installed. "
            "Please install it with `pip install diffusers`."
        )


    if library_name == "timm" and not is_timm_available():
        raise ImportError(
            "The library `timm` was specified, but it is not installed. Please install it with `pip install timm`."
        )

    if library_name is None:
        library_name = TasksManager.infer_library_from_model(
            model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
        )
        if library_name == "sentence_transformers" and not is_sentence_transformers_available():
            logger.warning(
                "The library name was inferred as `sentence_transformers`, which is not installed. "
                "Falling back to `transformers` to avoid breaking the export."
            )
            library_name = "transformers"
        elif library_name == "timm" and not is_timm_available():
            raise ImportError(
                "The library name was inferred as `timm`, which is not installed. "
                "Please install it with `pip install timm`."
            )
        elif library_name == "diffusers" and not is_diffusers_available():
            raise ImportError(
                "The library name was inferred as `diffusers`, which is not installed. "
                "Please install it with `pip install diffusers`."
            )

    # framework and dtype 
    if framework is None:
        framework = TasksManager.determine_framework(
            model_name_or_path, subfolder=subfolder, revision=revision, cache_dir=cache_dir, token=token
        )

    torch_dtype = None
    if framework == "pt":
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            torch_dtype = torch.float32

    loading_kwargs = {}
    is_mxfp4 = False
    if library_name == "transformers":
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        model_type = config.model_type

        is_mxfp4 = getattr(config, "quantization_config", {}).get("quant_method", None) == "mxfp4"
        # mxfp4 quantized model will be dequantized to bf16
        if is_mxfp4 and is_transformers_version(">=", "4.55") and Mxfp4Config is not None:
            torch_dtype = torch.float32 if model_type == "gpt_oss" else torch.bfloat16
            loading_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

        if model_type in SDPA_ARCHS_ONNX_EXPORT_NOT_SUPPORTED and is_transformers_version("<", "4.42"):
            loading_kwargs["attn_implementation"] = "eager"

        # For inference-driven export of model types not in TasksManager, force eager
        # attention so the masking code doesn't require cache_position during tracing.
        if export_by_inference and "attn_implementation" not in loading_kwargs:
            try:
                TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name="transformers")
            except KeyError:
                loading_kwargs["attn_implementation"] = "eager"

        # Only eager attention implementation returns attentions
        if model_kwargs is not None and model_kwargs.get("output_attentions", False):
            logger.warning(
                "The model is exported with `output_attentions=True`, which requires the attention implementation to be set to `eager`. "
                "Setting `attn_implementation='eager'` at loading time to ensure the attentions are returned by the model."
            )
            loading_kwargs["attn_implementation"] = "eager"


    original_task = task
    task = TasksManager.map_from_synonym(task)

    if task.endswith("-with-past") and monolith:
        task_non_past = task.replace("-with-past", "")
        raise ValueError(
            f"The task {task} is not compatible with the --monolith argument. Please either use"
            f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
        )

    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(
                model_name_or_path,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                library_name=library_name,
            )
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    with DisableCompileContextManager():
        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            framework=framework,
            torch_dtype=torch_dtype,
            device=device,
            library_name=library_name,
            **loading_kwargs,
        )

    needs_pad_token_id = task == "text-classification" and getattr(model.config, "pad_token_id", None) is None

    if needs_pad_token_id:
        if pad_token_id is not None:
            model.config.pad_token_id = pad_token_id
        else:
            tok = AutoTokenizer.from_pretrained(model_name_or_path)
            pad_token_id = getattr(tok, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                )
            model.config.pad_token_id = pad_token_id

    if hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type
    else:
        model_type = model.config.model_type

    # ensure gpt_oss models dtype is float32 (dequantized to bf16 by default leading to incompatible dtypes)
    if is_mxfp4 and model_type == "gpt_oss":
        model.to(torch.float32)

    try:
        _supported_tasks = TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
    except KeyError:
        _supported_tasks = []

    if (library_name != "diffusers" and task + "-with-past" in _supported_tasks):
        # Make -with-past the default if --task was not explicitly specified
        if original_task == "auto" and not monolith:
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )
            model.config.use_cache = False

    if task.endswith("with-past"):
        model.config.use_cache = True

    # The preprocessors are loaded as they may be useful to export the model. Notably, some of the static input shapes may be stored in the
    # preprocessors config.
    preprocessors = maybe_load_preprocessors(
        model_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
    )

    onnx_export_from_model(
        model=model,
        output=output,
        opset=opset,
        optimize=optimize,
        monolith=monolith,
        no_post_process=no_post_process,
        atol=atol,
        do_validation=do_validation,
        model_kwargs=model_kwargs,
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=fn_get_submodels,
        _variant=_variant,
        preprocessors=preprocessors,
        device=device,
        no_dynamic_axes=no_dynamic_axes,
        task=task,
        use_subprocess=use_subprocess,
        do_constant_folding=do_constant_folding,
        slim=slim,
        dynamo=dynamo,
        inference_kwargs=inference_kwargs,
        module_fixed_axis_fields=module_fixed_axis_fields,
        export_by_inference=export_by_inference,
        skip_random_generation=skip_random_generation,
        fixed_inputs=fixed_inputs,
        **kwargs_shapes,
    )

