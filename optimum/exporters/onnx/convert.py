if is_torch_available():
    import torch
    import torch.nn as nn
    from transformers.modeling_utils import PreTrainedModel

if is_diffusers_available():
    from diffusers import DiffusionPipeline, ModelMixin

from optimum.exporters.onnx.base import OnnxConfig
from pathlib import Path

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from optimum.exporters.onnx import validate_models_outputs, export_models
from optimum.exporters.tasks import TasksManager

try:
    from transformers.modeling_utils import get_parameter_dtype
except ImportError:
    # transformers>=5.0
    get_parameter_dtype = None

from optimum.exporters.onnx.utils import (
    PickableInferenceSession,
    _get_submodels_and_onnx_configs,
    _get_submodels_and_tensors_,
    recursive_to_device,
)

from optimum.utils import (
    DEFAULT_DUMMY_SHAPES,
    ONNX_WEIGHTS_NAME,
    TORCH_MINIMUM_VERSION,
    is_diffusers_available,
    is_onnxslim_available,
    is_torch_onnx_support_available,
    is_torch_version,
    is_transformers_version,
    logging,
)

from optimum.onnx.graph_transformations import check_and_save_model
from optimum.utils.save_utils import maybe_save_preprocessors

def onnx_export_from_model(
    model: PreTrainedModel | DiffusionPipeline,
    output: str | Path,
    opset: int | None = None,
    optimize: str | None = None,
    monolith: bool = False,
    no_post_process: bool = False,
    atol: float | None = None,
    do_validation: bool = True,
    model_kwargs: dict[str, Any] | None = None,
    custom_onnx_configs: dict[str, OnnxConfig] | None = None,
    fn_get_submodels: Callable | None = None,
    _variant: str = "default",
    preprocessors: list | None = None,
    device: str = "cpu",
    no_dynamic_axes: bool = False,
    task: str | None = None,
    use_subprocess: bool = False,
    do_constant_folding: bool = True,
    slim: bool = False,
    dynamo: bool = False,
    inference_kwargs: dict[str,Any] | None = None,
    module_fixed_axis_fields: dict[str, list[str]] | None = None,
    export_by_inference: bool = False,
    skip_random_generation: bool = False,
    **kwargs_shapes,
):
    TasksManager.standardize_model_attributes(model)

    if hasattr(model.config, "export_model_type"):
        model_type = model.config.export_model_type
    else:
        model_type = model.config.model_type

    library_name = TasksManager.infer_library_from_model(model)

    if task is not None:
        task = TasksManager.map_from_synonym(task)
    else:
        try:
            task = TasksManager._infer_task_from_model_or_model_class(model=model)
        except (ValueError, KeyError) as e:
            raise RuntimeError(
                f"The model task could not be automatically inferred in `onnx_export_from_model`. Please provide the argument `task` with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

        if (library_name != "diffusers"
            and task + "-with-past"
            in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx", library_name=library_name)
            and not monolith
        ):
            # -with-past is the default.
            task = task + "-with-past"
        logger.info(f"Automatic task detection to: {task}.")

    dtype = get_parameter_dtype(model) if isinstance(model, torch.nn.Module) and get_parameter_dtype else model.dtype
    if "bfloat16" in str(dtype):
        float_dtype = "bf16"
    elif "float16" in str(dtype):
        float_dtype = "fp16"
    else:
        float_dtype = "fp32"

    if task.startswith("text-generation") and model.config.is_encoder_decoder:
        raise ValueError(
            f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
            f"at https://github.com/huggingface/optimum, if --task was explicitly passed, make sure you selected the right task for the model,"
            f" referring to `optimum.exporters.tasks.TaskManager`'s `_TRANSFORMERS_TASKS_TO_MODEL_LOADERS`."
        )

    if library_name != "diffusers" and model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
        raise ValueError(
            f"{model_type} is not supported yet. Only {list(TasksManager._SUPPORTED_CLI_MODEL_TYPE.keys())} are supported. "
            f"If you want to support {model_type} please propose a PR or open up an issue."
        )

    output = Path('/dev/shm')
    if not output.exists():
        output.mkdir(parents=True)

    # inference model to trace input and output tensor shape
    models_and_inputs, models_and_outputs = _get_submodels_and_tensors_(
        model=model, 
        inference_kwargs=inference_kwargs,
        skip_random_generation=skip_random_generation,
    )

    onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=monolith,
        custom_onnx_configs=custom_onnx_configs if custom_onnx_configs is not None else {},
        custom_architecture=custom_architecture,
        float_dtype=float_dtype,
        fn_get_submodels=fn_get_submodels,
        preprocessors=preprocessors,
        _variant=_variant,
        library_name=library_name,
        model_kwargs=model_kwargs,
        models_and_inputs=models_and_inputs,
        models_and_outputs=models_and_outputs,
        module_fixed_axis_fields=module_fixed_axis_fields,
    )

    if library_name != "diffusers":
        # Ensure the requested opset is sufficient
        if opset is None:
            opset = onnx_config.DEFAULT_ONNX_OPSET
        elif opset < onnx_config.DEFAULT_ONNX_OPSET:
            logger.warning(
                f"Opset {opset} is lower than the recommended minimum opset ({onnx_config.DEFAULT_ONNX_OPSET}) to export {model_type}. "
                f"The ONNX export may fail or the exported model may be suboptimal."
            )
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace("-with-past", "")]

        if is_transformers_version(">=", "4.44.99") and is_transformers_version("<", "4.99"):
            misplaced_generation_parameters = model.config._get_non_default_generation_parameters()
            if (
                isinstance(model, GenerationMixin)
                and model.can_generate()
                and len(misplaced_generation_parameters) > 0
            ):
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(model.generation_config, param_name, param_value)
                    setattr(model.config, param_name, None)

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            # since v4.41.0 an exceptions will be raised when saving a generation config considered invalid
            # https://github.com/huggingface/transformers/blob/v4.41.0/src/transformers/generation/configuration_utils.py#L697
            try:
                generation_config.save_pretrained(output)
            except Exception as exception:
                logger.warning(f"The generation config is invalid and will not be saved : {exception}")

        model_name_or_path = model.config._name_or_path
        maybe_save_preprocessors(model_name_or_path, output)

        onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs]
    else:
        # save the subcomponent configuration
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        onnx_files_subpaths = [os.path.join(name_dir, ONNX_WEIGHTS_NAME) for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        tokenizer_3 = getattr(model, "tokenizer_3", None)
        if tokenizer_3 is not None:
            tokenizer_3.save_pretrained(output.joinpath("tokenizer_3"))

        model.save_config(output)

    if float_dtype == "bf16":
        logger.warning(
            f"Exporting the model {model.__class__.__name__} in bfloat16 float dtype. After the export, ONNX Runtime InferenceSession with CPU/CUDA execution provider likely does not implement all operators for the bfloat16 data type, and the loading is likely to fail."
        )

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=opset,
        output_dir=output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=device,
        dtype=float_dtype,
        no_dynamic_axes=no_dynamic_axes,
        do_constant_folding=do_constant_folding,
        dynamo=dynamo,
        model_kwargs=model_kwargs,
        export_by_inference=export_by_inference,
    )

    if models_and_outputs is not None:
        import json
        output_dir = os.path.join(output, "io_binding")
        os.makedirs(output_dir, exist_ok=True)
        
        for module_name, dummy_outputs in models_and_outputs.items():
            # convert tuple -> list for json
            serializable = {
                name: list(shape)
                for name, shape in dummy_outputs.items()
            }

            file_path = os.path.join(output_dir, f"{module_name}_outputs.json")

            with open(file_path, "w") as f:
                json.dump(serializable, f, indent=4)
            print(f"Saved: {file_path}")


    if optimize is not None:
        from optimum.onnxruntime import AutoOptimizationConfig, ORTOptimizer

        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)

        optimization_config.disable_shape_inference = True
        optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix="")

    if slim:
        if not is_onnxslim_available():
            raise ImportError("The pip package `onnxslim` is required to optimize onnx models.")

        from onnxslim import slim

        for subpath in onnx_files_subpaths:
            file_path = os.path.join(output, subpath)
            slimmed_model = slim(file_path)
            check_and_save_model(slimmed_model, file_path)

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating diffusion separately is quite ugly
    if not no_post_process and library_name != "diffusers":
        try:
            logger.info("Post-processing the exported models...")
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise RuntimeError(
                "The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process"
            ) from e

    if library_name == "diffusers":
        # TODO: fix Can't pickle local object 'get_stable_diffusion_models_for_export.<locals>.<lambda>'
        use_subprocess = False
    elif model_type in UNPICKABLE_ARCHS:
        # Pickling is bugged for nn.utils.weight_norm: https://github.com/pytorch/pytorch/issues/102983
        # TODO: fix "Cowardly refusing to serialize non-leaf tensor" error for wav2vec2-conformer
        use_subprocess = False

    if device == "cpu":
        # Using multiprocessing for validation is useful only on CUDA EP that leaks memory.
        use_subprocess = False

    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                atol=atol,
                output_dir=output,
                onnx_files_subpaths=onnx_files_subpaths,
                input_shapes=input_shapes,
                device=device,
                use_subprocess=use_subprocess,
                model_kwargs=model_kwargs,
            )
            logger.info(f"The ONNX export succeeded and the exported model was saved at: {output.as_posix()}")
        except ShapeError:
            raise
        except AtolError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred during validation, but the model was saved nonetheless at {output.as_posix()}"
            ) from e


