from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForMaskedLM,
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
    ORTModelForQuestionAnswering,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def _on_the_fly_export(
    cls,
    model_id: str | Path,
    config: "PretrainedConfig | None",
    inference_kwargs: "dict[str, Any] | None",
    module_fixed_axis_fields: "dict[str, list[str]] | None",
    skip_random_generation: bool,
    subfolder: str,
    revision: str,
    force_download: bool,
    local_files_only: bool,
    trust_remote_code: bool,
    cache_dir: str,
    token: "bool | str | None",
    provider: str,
    providers: "list | None",
    provider_options,
    session_options,
    use_io_binding: "bool | None",
    extra_from_pretrained_kwargs: dict,
):
    """Shared inference-driven export logic for all OnTheFly ORT classes."""
    import torch
    from transformers import AutoConfig
    from optimum.exporters.tasks import TasksManager
    from optimum.exporters.onnx.convert import export_models
    from optimum.utils.save_utils import maybe_save_preprocessors

    from inference_driven_model_compiler.optimum.exporters.onnx.utils import (
        trace_model_shapes, generate_config_dim,
    )
    from inference_driven_model_compiler.optimum.exporters.onnx.model_configs import DummyOnnxConfig

    # 1. Config
    if config is None:
        config = AutoConfig.from_pretrained(
            model_id, subfolder=subfolder, revision=revision,
            cache_dir=cache_dir, token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

    # 2. Load PyTorch model
    task = TasksManager._infer_task_from_model_or_model_class(model_class=cls.auto_model_class)
    pytorch_model = TasksManager.get_model_from_task(
        task, model_id,
        subfolder=subfolder, revision=revision,
        cache_dir=cache_dir, token=token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework="pt",
    )
    pytorch_model.eval()

    # 3. Trace shapes across multiple varied runs to derive dynamic axes
    n_trials = extra_from_pretrained_kwargs.pop("n_trials", 3)
    inputs, outputs, dynamic_axes = trace_model_shapes(
        pytorch_model,
        inference_kwargs or {},
        skip_random_generation=skip_random_generation,
        n_trials=n_trials,
    )

    # 4. Build DummyOnnxConfig (dynamic_axes already empirically derived)
    dim_names = (module_fixed_axis_fields or {}).get("transformer", [])
    config_dim = generate_config_dim(pytorch_model, dim_names)
    onnx_cfg = DummyOnnxConfig(
        config=pytorch_model.config,
        task="backbone",
        model_inputs=inputs,
        model_outputs=outputs,
        config_dim=config_dim,
        dynamic_axes=dynamic_axes,
    )

    # 5. Export — use a unique temp dir to avoid collisions across runs
    import tempfile, uuid
    save_dir = Path(tempfile.gettempdir()) / f"on_the_fly_{uuid.uuid4().hex[:8]}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # For encoder-decoder models, export only the encoder submodule so that
    # the ONNX graph requires only encoder inputs (input_ids, attention_mask).
    is_enc_dec = getattr(getattr(pytorch_model, "config", None), "is_encoder_decoder", False)
    export_submodel = pytorch_model.encoder if (is_enc_dec and hasattr(pytorch_model, "encoder")) else pytorch_model
    export_models(
        models_and_onnx_configs={"transformer": (export_submodel, onnx_cfg)},
        opset=onnx_cfg.DEFAULT_ONNX_OPSET,
        output_dir=save_dir,
        output_names=["transformer.onnx"],
        # We already derive dynamic axes empirically via multi-trial tracing,
        # so skip optimum's re-derivation (its KV flattening uses a different
        # naming scheme and would fail validation).
        disable_dynamic_axes_fix=True,
    )
    maybe_save_preprocessors(model_id, save_dir, src_subfolder=subfolder)

    # 6. Load ORT session from the unique temp dir
    has_kv = onnx_cfg.use_past
    return cls._from_pretrained(
        save_dir, config,
        file_name="transformer.onnx",
        model_save_dir=save_dir,
        provider=provider,
        providers=providers,
        provider_options=provider_options,
        session_options=session_options,
        use_io_binding=use_io_binding,
        subfolder="",
        revision=revision,
        force_download=force_download,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        token=token,
        **({"use_cache": has_kv} if has_kv else {}),
        **extra_from_pretrained_kwargs,
    )


class _OnTheFlyORTMixin:
    """Mixin: adds inference_kwargs / module_fixed_axis_fields API to any ORT model class.

    When export_by_inference=True, runs actual inference to trace tensor shapes,
    exports to ONNX, and loads the result — without touching optimum-onnx source.
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
        n_trials: int = 3,          # number of inference passes for axis detection
        # hub options
        subfolder: str = "",
        revision: str = "main",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        token: "bool | str | None" = None,
        # session options
        provider: str = "CPUExecutionProvider",
        providers=None,
        provider_options=None,
        session_options=None,
        use_io_binding: "bool | None" = None,
        **kwargs,
    ):
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        if cache_dir is None:
            cache_dir = HUGGINGFACE_HUB_CACHE

        if export and export_by_inference:
            return _on_the_fly_export(
                cls=cls,
                model_id=model_id,
                config=config,
                inference_kwargs=inference_kwargs,
                module_fixed_axis_fields=module_fixed_axis_fields,
                skip_random_generation=skip_random_generation,
                subfolder=subfolder,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
                token=token,
                provider=provider,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
                use_io_binding=use_io_binding,
                extra_from_pretrained_kwargs={**kwargs, "n_trials": n_trials},
            )

        # Standard path (loading existing ONNX or normal export)
        return super().from_pretrained(
            model_id,
            config=config,
            export=export,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            token=token,
            provider=provider,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
            use_io_binding=use_io_binding,
            **kwargs,
        )


class OnTheFlyORTModelForFeatureExtraction(_OnTheFlyORTMixin, ORTModelForFeatureExtraction):
    """Inference-driven ONNX export for feature-extraction (BERT, ViT, CLIP, T5-enc, …)."""


class OnTheFlyORTModelForMaskedLM(_OnTheFlyORTMixin, ORTModelForMaskedLM):
    """Inference-driven ONNX export for masked language models (BERT, RoBERTa, …)."""


class OnTheFlyORTModelForSequenceClassification(_OnTheFlyORTMixin, ORTModelForSequenceClassification):
    """Inference-driven ONNX export for sequence classification."""


class OnTheFlyORTModelForTokenClassification(_OnTheFlyORTMixin, ORTModelForTokenClassification):
    """Inference-driven ONNX export for token classification / NER."""


class OnTheFlyORTModelForQuestionAnswering(_OnTheFlyORTMixin, ORTModelForQuestionAnswering):
    """Inference-driven ONNX export for extractive question answering."""
