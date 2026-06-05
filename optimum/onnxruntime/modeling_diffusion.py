from __future__ import annotations

import importlib
import inspect
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import SchedulerMixin
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils.constants import CONFIG_NAME
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.utils import http_user_agent

from onnxruntime import InferenceSession, SessionOptions
from optimum.exporters.onnx import main_export
from optimum.onnxruntime.base import ORTParentMixin, ORTSessionMixin
from optimum.onnxruntime.utils import get_device_for_provider, prepare_providers_and_provider_options
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
    DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
    ONNX_WEIGHTS_NAME,
    is_diffusers_version,
)

from .utils import load_shapes_as_torch_size

if is_diffusers_version(">=", "0.25.0"):
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
else:
    from diffusers.models.vae import DiagonalGaussianDistribution  # type: ignore

if is_diffusers_version(">=", "0.35.0"):
    from diffusers.models.cache_utils import CacheMixin
else:
    CacheMixin = object

logger = logging.getLogger(__name__)


class ORTModelMixin(ORTSessionMixin, ConfigMixin, CacheMixin):
    config_name: str = CONFIG_NAME

    def __init__(
        self,
        session: InferenceSession,
        parent: "ORTDiffusionPipeline",
        use_io_binding: bool | None = None,
    ):
        self.initialize_ort_attributes(session, use_io_binding=use_io_binding)
        self.parent = parent

        config_file_path = Path(session._model_path).parent / self.config_name
        if not config_file_path.is_file():
            raise ValueError(f"Configuration file for {self.__class__.__name__} not found at {config_file_path}")
        config_dict = self._dict_from_json_file(config_file_path)
        self.register_to_config(**config_dict)

        self.io_binding_file = None

    def set_io_binding_file(self, filename: str):
        self.io_binding_file = filename

    def save_pretrained(self, save_directory: str | Path):
        self.save_session(save_directory)
        self.save_config(save_directory)

    def named_modules(self):
        # diffusers >= 0.35.0 calls named_modules for KV-cache hooks — not applicable to ORT
        yield from []


class ORTUnet(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self.config, "time_cond_proj_dim"):
            self.register_to_config(time_cond_proj_dim=None)

        if len(self.input_shapes["timestep"]) > 0:
            logger.warning(
                "The exported unet onnx model expects a non-scalar timestep input. "
                "Please re-export the pipeline with a newer version of optimum and diffusers."
            )

    def forward(
        self,
        sample: np.ndarray | torch.Tensor,
        timestep: np.ndarray | torch.Tensor,
        encoder_hidden_states: np.ndarray | torch.Tensor,
        timestep_cond: np.ndarray | torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(sample, torch.Tensor)

        if len(self.input_shapes["timestep"]) > 0:
            timestep = timestep.unsqueeze(0)

        model_inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep_cond": timestep_cond,
            **(cross_attention_kwargs or {}),
            **(added_cond_kwargs or {}),
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            known_output_shapes["out_sample"] = sample.shape
            known_output_buffers = None
            if "LatentConsistencyModel" not in self.parent.__class__.__name__:
                known_output_buffers = {"out_sample": sample}
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                known_output_shapes=known_output_shapes,
                known_output_buffers=known_output_buffers,
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        model_outputs["sample"] = model_outputs.pop("out_sample")

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTTransformer(ORTModelMixin):
    def forward(
        self,
        hidden_states: np.ndarray | torch.Tensor,
        encoder_hidden_states: np.ndarray | torch.Tensor,
        timestep: np.ndarray | torch.Tensor,
        pooled_projections: np.ndarray | torch.Tensor | None = None,
        guidance: np.ndarray | torch.Tensor | None = None,
        txt_ids: np.ndarray | torch.Tensor | None = None,
        img_ids: np.ndarray | torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(hidden_states, torch.Tensor)

        model_inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "guidance": guidance,
            "txt_ids": txt_ids,
            "img_ids": img_ids,
            **(joint_attention_kwargs or {}),
            **(attention_kwargs or {}),
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            known_output_shapes["out_hidden_states"] = hidden_states.shape
            known_output_buffers = None
            if "Flux" not in self.parent.__class__.__name__:
                known_output_buffers = {"out_hidden_states": hidden_states}
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs,
                known_output_shapes=known_output_shapes,
                known_output_buffers=known_output_buffers,
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTTextEncoder(ORTModelMixin):
    def forward(
        self,
        input_ids: np.ndarray | torch.Tensor,
        attention_mask: np.ndarray | torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(input_ids, torch.Tensor)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes, known_output_buffers=None
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if output_hidden_states:
            model_outputs["hidden_states"] = []
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs["hidden_states"].append(model_outputs.pop(f"hidden_states.{i}"))
            model_outputs["hidden_states"].append(model_outputs.get("last_hidden_state"))
        else:
            num_layers = self.num_hidden_layers if hasattr(self, "num_hidden_layers") else self.num_decoder_layers
            for i in range(num_layers):
                model_outputs.pop(f"hidden_states.{i}", None)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVaeEncoder(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.config, "scaling_factor"):
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        sample: np.ndarray | torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(sample, torch.Tensor)
        model_inputs = {"sample": sample}

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes, known_output_buffers=None
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if "latent_sample" in model_outputs:
            model_outputs["latents"] = model_outputs.pop("latent_sample")
        if "latent_parameters" in model_outputs:
            model_outputs["latent_dist"] = DiagonalGaussianDistribution(
                parameters=model_outputs.pop("latent_parameters")
            )

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVaeDecoder(ORTModelMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.config, "scaling_factor") and hasattr(self.config, "block_out_channels"):
            self.register_to_config(scaling_factor=2 ** (len(self.config.block_out_channels) - 1))

    def forward(
        self,
        latent_sample: np.ndarray | torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ):
        use_torch = isinstance(latent_sample, torch.Tensor)
        model_inputs = {"latent_sample": latent_sample}

        if self.use_io_binding:
            known_output_shapes = load_shapes_as_torch_size(self.io_binding_file)
            output_shapes, output_buffers = self._prepare_io_binding(
                model_inputs, known_output_shapes=known_output_shapes, known_output_buffers=None
            )
            if self.device.type == "cpu":
                self.session.run_with_iobinding(self._io_binding)
            else:
                self._io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self._io_binding)
                self._io_binding.synchronize_outputs()
            model_outputs = {name: output_buffers[name].view(output_shapes[name]) for name in self.output_names}
        else:
            onnx_inputs = self._prepare_onnx_inputs(use_torch, model_inputs)
            onnx_outputs = self.session.run(None, onnx_inputs)
            model_outputs = self._prepare_onnx_outputs(use_torch, onnx_outputs)

        if not return_dict:
            return tuple(model_outputs.values())
        return ModelOutput(**model_outputs)


class ORTVae(ORTParentMixin):
    def __init__(self, encoder: ORTVaeEncoder | None = None, decoder: ORTVaeDecoder | None = None):
        self.encoder = encoder
        self.decoder = decoder
        self.initialize_ort_attributes(parts=list(filter(None, {self.encoder, self.decoder})))

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    @property
    def config(self):
        return self.decoder.config


def _make_ort_pipeline_class(diffusers_class: type) -> type:
    """Dynamically create an ORT pipeline class for any diffusers pipeline.

    Returns a class that inherits from both ORTDiffusionPipeline and the given
    diffusers pipeline class, so it runs inference via ONNX Runtime while
    keeping the full diffusers pipeline API (schedulers, tokenizers, etc.).

    No manual subclass is needed when a new diffusers pipeline is released.
    """
    return type(
        f"ORT{diffusers_class.__name__}",
        (ORTDiffusionPipeline, diffusers_class),
        {"auto_model_class": diffusers_class, "task": "auto"},
    )


class ORTDiffusionPipeline(ORTParentMixin, DiffusionPipeline):
    """Generic ONNX Runtime pipeline for any diffusers DiffusionPipeline.

    Export any diffusers pipeline to ONNX and run it with ONNX Runtime — no
    model-specific subclass required.  A compatible class is created on the fly
    from the pipeline's ``_class_name`` config entry.

    Usage::

        pipe = ORTDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            export=True,
        )
        image = pipe("a photo of an astronaut").images[0]
    """

    config_name = DIFFUSION_PIPELINE_CONFIG_FILE_NAME
    task = "auto"
    library = "diffusers"
    auto_model_class = DiffusionPipeline

    def __init__(
        self,
        *,
        unet_session: InferenceSession | None = None,
        transformer_session: InferenceSession | None = None,
        vae_decoder_session: InferenceSession | None = None,
        vae_encoder_session: InferenceSession | None = None,
        text_encoder_session: InferenceSession | None = None,
        text_encoder_2_session: InferenceSession | None = None,
        text_encoder_3_session: InferenceSession | None = None,
        scheduler: SchedulerMixin | None = None,
        tokenizer: CLIPTokenizer | None = None,
        tokenizer_2: CLIPTokenizer | None = None,
        tokenizer_3: CLIPTokenizer | None = None,
        feature_extractor: CLIPFeatureExtractor | None = None,
        force_zeros_for_empty_prompt: bool = True,
        requires_aesthetics_score: bool = False,
        add_watermarker: bool | None = None,
        use_io_binding: bool | None = None,
        model_save_dir: str | Path | None = None,
        **kwargs,
    ):
        self.unet = ORTUnet(unet_session, self, use_io_binding) if unet_session is not None else None
        self.transformer = ORTTransformer(transformer_session, self, use_io_binding) if transformer_session is not None else None
        self.text_encoder = ORTTextEncoder(text_encoder_session, self, use_io_binding) if text_encoder_session is not None else None
        self.text_encoder_2 = ORTTextEncoder(text_encoder_2_session, self, use_io_binding) if text_encoder_2_session is not None else None
        self.text_encoder_3 = ORTTextEncoder(text_encoder_3_session, self, use_io_binding) if text_encoder_3_session is not None else None
        self.vae_encoder = ORTVaeEncoder(vae_encoder_session, self, use_io_binding) if vae_encoder_session is not None else None
        self.vae_decoder = ORTVaeDecoder(vae_decoder_session, self, use_io_binding) if vae_decoder_session is not None else None

        super().initialize_ort_attributes(
            parts=list(filter(None, {
                self.unet, self.transformer,
                self.vae_encoder, self.vae_decoder,
                self.text_encoder, self.text_encoder_2, self.text_encoder_3,
            }))
        )

        self.vae = (
            ORTVae(self.vae_encoder, self.vae_decoder)
            if self.vae_encoder is not None or self.vae_decoder is not None
            else None
        )

        self.image_encoder = kwargs.pop("image_encoder", None)
        self.safety_checker = kwargs.pop("safety_checker", None)

        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.feature_extractor = feature_extractor

        all_pipeline_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "tokenizer_3": self.tokenizer_3,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }
        diffusers_pipeline_args = {
            k: v for k, v in all_pipeline_init_args.items()
            if k in inspect.signature(self.auto_model_class).parameters
        }
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        self.model_save_dir = model_save_dir

    @property
    def components(self) -> dict[str, Any]:
        components = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "text_encoder_3": self.text_encoder_3,
            "safety_checker": self.safety_checker,
            "image_encoder": self.image_encoder,
        }
        return {k: v for k, v in components.items() if v is not None}

    def to(self, device: torch.device | str | int):
        for component in self.components.values():
            if isinstance(component, (ORTSessionMixin, ORTParentMixin)):
                component.to(device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        export: bool | None = None,
        provider: str = "CPUExecutionProvider",
        providers: Sequence[str] | None = None,
        provider_options: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
        session_options: SessionOptions | None = None,
        use_io_binding: bool | None = None,
        **kwargs,
    ):
        providers, provider_options = prepare_providers_and_provider_options(
            provider=provider, providers=providers, provider_options=provider_options
        )

        hf_api = HfApi(user_agent=http_user_agent())
        hub_kwargs = {
            "force_download": kwargs.get("force_download", False),
            "resume_download": kwargs.get("resume_download"),
            "local_files_only": kwargs.get("local_files_only", False),
            "cache_dir": kwargs.get("cache_dir"),
            "revision": kwargs.get("revision"),
            "proxies": kwargs.get("proxies"),
            "token": kwargs.get("token"),
        }

        config = cls.load_config(model_name_or_path, **hub_kwargs)
        config = config[0] if isinstance(config, tuple) else config

        model_save_path = Path(model_name_or_path)

        # auto-detect whether export is needed
        if export is None:
            if "unet" in config and config["unet"] is not None:
                relative_file_path = Path(DIFFUSION_MODEL_UNET_SUBFOLDER) / ONNX_WEIGHTS_NAME
            elif "transformer" in config and config["transformer"] is not None:
                relative_file_path = Path(DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER) / ONNX_WEIGHTS_NAME
            else:
                raise ValueError(
                    "Neither 'unet' nor 'transformer' found in pipeline config. "
                    "Set export=True or export=False explicitly."
                )
            absolute_file_path = model_save_path / relative_file_path
            export = not (
                absolute_file_path.is_file()
                or hf_api.file_exists(
                    repo_id=str(model_name_or_path),
                    filename=str(relative_file_path),
                    revision=hub_kwargs.get("revision"),
                    token=hub_kwargs.get("token"),
                )
            )

        if export:
            model_save_path = Path("/dev/shm")

            torch_dtype = kwargs.pop("torch_dtype", None)
            if torch_dtype is not None:
                if torch_dtype == torch.float16:
                    kwargs["dtype"] = "fp16"
                elif torch_dtype == torch.float32:
                    kwargs["dtype"] = "fp32"
                else:
                    raise ValueError(f"Unsupported torch_dtype for export: {torch_dtype}")

            export_kwargs = {
                "slim": kwargs.pop("slim", False),
                "dtype": kwargs.pop("dtype", None),
                "device": get_device_for_provider(provider, {}).type,
                "no_dynamic_axes": kwargs.pop("no_dynamic_axes", False),
            }
            main_export(
                model_name_or_path=str(model_name_or_path),
                output=model_save_path,
                no_post_process=True,
                do_validation=False,
                task="auto",
                **{k: v for k, v in export_kwargs.items() if v is not None},
                **hub_kwargs,
            )

        # download model from hub if it's not a local directory
        if not model_save_path.is_dir():
            all_components = {key for key in config if not key.startswith("_")} | {"vae_encoder", "vae_decoder"}
            allow_patterns = {os.path.join(component, "*") for component in all_components}
            allow_patterns.update({
                ONNX_WEIGHTS_NAME, DIFFUSION_PIPELINE_CONFIG_FILE_NAME,
                SCHEDULER_CONFIG_NAME, CONFIG_NAME,
            })
            model_save_folder = hf_api.snapshot_download(
                repo_id=str(model_name_or_path),
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin", "*.xml"],
                **hub_kwargs,
            )
            model_save_path = Path(model_save_folder)

        model_paths = {
            "unet":          model_save_path / DIFFUSION_MODEL_UNET_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "transformer":   model_save_path / DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "vae_encoder":   model_save_path / DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "vae_decoder":   model_save_path / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder":  model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder_2": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER / ONNX_WEIGHTS_NAME,
            "text_encoder_3": model_save_path / DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER / ONNX_WEIGHTS_NAME,
        }

        sessions = {}
        models = {}
        for model_key, path in model_paths.items():
            if kwargs.get(model_key) is not None:
                models[model_key] = kwargs.pop(model_key)
            elif kwargs.get(f"{model_key}_session") is not None:
                sessions[f"{model_key}_session"] = kwargs.pop(f"{model_key}_session")
            elif path.is_file():
                sessions[f"{model_key}_session"] = InferenceSession(
                    path,
                    providers=providers,
                    provider_options=provider_options,
                    sess_options=session_options,
                )

        submodels = {}
        for submodel in {"scheduler", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"}:
            if kwargs.get(submodel) is not None:
                submodels[submodel] = kwargs.pop(submodel)
            elif config.get(submodel, (None, None))[0] is not None:
                library_name, library_classes = config.get(submodel)
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                if (model_save_path / submodel).is_dir():
                    submodels[submodel] = class_obj.from_pretrained(model_save_path / submodel)
                else:
                    submodels[submodel] = class_obj.from_pretrained(model_save_path)

        # Resolve the concrete pipeline class dynamically — no hardcoded mapping needed.
        # For any new diffusers pipeline, the right ORT class is created on the fly.
        if cls is ORTDiffusionPipeline:
            pipeline_class_name = config["_class_name"]
            diffusers_class = getattr(diffusers, pipeline_class_name, None)
            if diffusers_class is None:
                raise ValueError(
                    f"Pipeline class '{pipeline_class_name}' not found in diffusers. "
                    f"Make sure diffusers is up to date."
                )
            ort_pipeline_class = _make_ort_pipeline_class(diffusers_class)
        else:
            ort_pipeline_class = cls

        ort_pipeline = ort_pipeline_class(
            **sessions,
            **submodels,
            **models,
            use_io_binding=use_io_binding,
            **kwargs,
        )

        ort_pipeline.register_to_config(**config)
        ort_pipeline.register_to_config(_name_or_path=config.get("_name_or_path", str(model_name_or_path)))

        # Wire up IO binding shape files for each component
        io_binding_dir = model_save_path / "io_binding"
        for key, comp in ort_pipeline.components.items():
            if key == "vae":
                if comp.encoder is not None:
                    comp.encoder.set_io_binding_file(str(io_binding_dir / "vae_encoder_outputs.json"))
                if comp.decoder is not None:
                    comp.decoder.set_io_binding_file(str(io_binding_dir / "vae_decoder_outputs.json"))
            else:
                comp.set_io_binding_file(str(io_binding_dir / f"{key}_outputs.json"))

        return ort_pipeline

    def save_pretrained(self, save_directory: str | Path, push_to_hub: bool = False, **kwargs):
        model_save_path = Path(save_directory)
        model_save_path.mkdir(parents=True, exist_ok=True)

        self.save_config(model_save_path)
        self.scheduler.save_pretrained(model_save_path / "scheduler")

        for attr, subfolder in [
            ("unet",          DIFFUSION_MODEL_UNET_SUBFOLDER),
            ("transformer",   DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER),
            ("vae_encoder",   DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER),
            ("vae_decoder",   DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER),
            ("text_encoder",  DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER),
            ("text_encoder_2", DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER),
            ("text_encoder_3", DIFFUSION_MODEL_TEXT_ENCODER_3_SUBFOLDER),
        ]:
            component = getattr(self, attr, None)
            if component is not None:
                component.save_pretrained(model_save_path / subfolder)

        for attr in ("image_encoder", "safety_checker", "tokenizer", "tokenizer_2", "tokenizer_3", "feature_extractor"):
            component = getattr(self, attr, None)
            if component is not None:
                component.save_pretrained(model_save_path / attr)
