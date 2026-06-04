import torch

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.tasks import TasksManager
from optimum.onnxruntime import InferenceSession, SessionOptions
from optimum.onnxruntime.constants import ONNX_FILE_PATTERN

from transformers import AutoModelForCausalLM, GenerationConfig

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

if TYPE_CHECKING:
    from transformers import PretrainedConfig
from collections.abc import Sequence
from optimum.exporters.onnx import main_export

class OnTheFlyORTModelForCausalLM(ORTModelForCausalLM):
    @classmethod
    def _from_pretrained(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # file options
        file_name: str | None = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        session_options: SessionOptions | None = None,
        # inference options
        use_cache: bool = True,
        use_io_binding: bool | None = None,
        generation_config: GenerationConfig | None = None,
        dtype: torch.dtype = torch.float32,
        # other arguments
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        # export options
        inference_kwargs: dict[str, Any] | None = None,
        module_fixed_axis_fields: dict[str, Any] | None=None,
        export_by_inference: bool = False,
        skip_random_generation: bool = False,
    ) -> ORTModelForCausalLM:
        
        onnx_files = find_files_matching_pattern(
            model_id,
            ONNX_FILE_PATTERN,
            glob_pattern="**/*.onnx",
            subfolder=subfolder,
            revision=revision,
            token=token,
        )
        if len(onnx_files) == 0:
            raise FileNotFoundError(f"Could not find any ONNX model file in {model_id}")
        if Path(model_id).is_dir():
            onnx_files = [f.relative_to(model_id) for f in onnx_files]

        file_path = cls._infer_file_path(
            ONNX_FILE_PATTERN,
            onnx_files=onnx_files,
            standard_file_name=ONNX_WEIGHTS_NAME,
            target_file_name=file_name,
        )

        model_path = cls._cached_file(
            model_id,
            filename=file_path.name,
            subfolder=file_path.parent.as_posix(),
            force_download=force_download,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance,
        # in which case we want to keep it instead.
        if model_save_dir is None:
            model_save_dir = model_path.parent

        # Important: for encoder-decoder models used with CausalLM, we need to set the is_decoder flag to True
        # and the is_encoder_decoder flag to False. This is needed for the model to work correctly with generation logic.
        config.use_cache = use_cache
        if hasattr(config, "is_decoder"):
            config.is_decoder = True
        if hasattr(config, "is_encoder_decoder"):
            config.is_encoder_decoder = False
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "onnxruntime"

        if generation_config is None:
            try:
                generation_config = GenerationConfig.from_pretrained(
                    model_id,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
            except OSError:
                logger.info("Generation config file not found, creating a new one from model config.")
                generation_config = GenerationConfig.from_model_config(config)

        generation_config.use_cache = use_cache
        if hasattr(generation_config, "cache_implementation"):
            generation_config.cache_implementation = None

        if is_transformers_version(">=", "4.45.0") and is_transformers_version("<", "4.99"):
            misplaced_generation_parameters = config._get_non_default_generation_parameters()
            if len(misplaced_generation_parameters) > 0:
                logger.warning(
                    "Moving the following attributes in the config to the generation config: "
                    f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                    "generation parameters in the model config, as opposed to in the generation config.",
                )
                for param_name, param_value in misplaced_generation_parameters.items():
                    setattr(generation_config, param_name, param_value)
                    setattr(config, param_name, None)

        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )
        session = InferenceSession(
            model_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )

        return cls(
            config=config,
            session=session,
            use_io_binding=use_io_binding,
            generation_config=generation_config,
            model_save_dir=model_save_dir,
        )

    @classmethod
    def _export(
        cls,
        model_id: str | Path,
        config: PretrainedConfig,
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token: bool | str | None = None,
        # inference options
        use_cache: bool = True,
        # export options
        inference_kwargs: dict[str, Any] | None = None,
        module_fixed_axis_fields: dict[str, Any] | None=None,
        export_by_inference: bool = False,
        skip_random_generation: bool = False,
        **kwargs,
    ) -> ORTModelForCausalLM:
        # this is guaranteed to work since we it uses a mapping from model classes to task names
        # instead of relying on the hub metadata or the model configuration
        task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
        if use_cache:
            task += "-with-past"

        if kwargs.get("task") is not None:
            raise ValueError(
                f"The `task` argument is not needed when exporting a model with `{cls.__name__}`. "
                f"The `task` is automatically inferred from the class as `{task}`."
            )

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            task=task,
            do_validation=False,
            no_post_process=False,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            # export options
            inference_kwargs=inference_kwargs,
            module_fixed_axis_fields=module_fix_axis_fields,
            export_by_inference=export_by_inference,
            skip_random_generation=skip_random_generation,
        )
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            model_save_dir=save_dir,
            **kwargs,
        )