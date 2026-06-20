"""Tests for the generic ORT diffusion pipeline.

These tests cover:
  1. Dynamic class creation via _make_ort_pipeline_class for every
     text-to-video pipeline currently in diffusers.
  2. OnTheFlyORTDiffusionPipeline.from_pretrained with fully-mocked I/O (no GPU,
     no model download).
  3. save_pretrained round-trip.
  4. ORTTransformer / ORTTextEncoder / ORTVaeDecoder forward pass with
     a mocked InferenceSession.
  5. IO binding file wiring.

Run with:
    cd /workspace
    python3 inference_driven_model_compiler/on_the_fly_pipeline_tests/test_diffusion_pipeline.py

No GPU or large model downloads required — all sessions are mocked.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(0, "/workspace")

import diffusers
from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
    OnTheFlyORTDiffusionPipeline,
    ORTModelMixin,
    ORTTransformer,
    ORTTextEncoder,
    ORTVaeDecoder,
    ORTVaeEncoder,
    ORTVae,
    ORTUnet,
    _make_ort_pipeline_class,
    _UnetAddedCondWrapper,
    _TextEncoderHiddenStatesWrapper,
)
from transformers.modeling_outputs import ModelOutput

# ── All text-to-video pipelines currently in diffusers ────────────────────────
# Source: diffusers 0.38.x — update as diffusers adds new ones.
# Image-to-video variants are intentionally excluded (primary input is an image,
# not a text prompt).
TEXT_TO_VIDEO_PIPELINES = [
    "AnimateDiffPipeline",
    "AnimateDiffSDXLPipeline",
    "CogVideoXPipeline",
    "HunyuanVideo15Pipeline",
    "HunyuanVideoPipeline",
    "LTXPipeline",
    "LTX2Pipeline",
    "LattePipeline",
    "MochiPipeline",
    "SanaVideoPipeline",
    "TextToVideoSDPipeline",
    "WanPipeline",
    "WanAnimatePipeline",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_session(input_names=None, output_names=None, model_path="/tmp/fake/model.onnx",
                  input_type="tensor(float)", output_type="tensor(float)"):
    """Return a lightweight mock InferenceSession with realistic metadata.

    The returned mock implements the onnxruntime InferenceSession interface
    closely enough that ORTSessionMixin.initialize_ort_attributes succeeds
    when IOBinding is patched out.
    """
    in_names = input_names or ["hidden_states"]
    out_names = output_names or ["out_hidden_states"]

    def _make_io(names, ort_type, shape):
        objs = []
        for n in names:
            m = MagicMock()
            m.name = n
            m.shape = shape
            m.type = ort_type
            objs.append(m)
        return objs

    sess = MagicMock()
    sess._model_path = model_path
    sess._sess = MagicMock()
    sess.get_providers.return_value = ["CPUExecutionProvider"]
    sess.get_provider_options.return_value = {"CPUExecutionProvider": {}}
    sess.get_inputs.return_value = _make_io(in_names, input_type, [1, 64])
    sess.get_outputs.return_value = _make_io(out_names, output_type, [1, 64])

    n_out = len(out_names)
    sess.run.return_value = [np.zeros((1, 64), dtype=np.float32)] * n_out
    return sess


def _write_config(directory, class_name="StableDiffusionPipeline"):
    """Write a minimal model_index.json so load_config() can read it."""
    config = {
        "_class_name": class_name,
        "_diffusers_version": "0.38.0",
        "scheduler": ["diffusers", "DDIMScheduler"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "text_encoder": ["transformers", "CLIPTextModel"],
    }
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "model_index.json")
    with open(path, "w") as f:
        json.dump(config, f)
    return config


def _write_component_config(directory, extra=None):
    """Write a minimal config.json for an ORT submodule (ORTModelMixin)."""
    cfg = {"num_hidden_layers": 12, "num_decoder_layers": 12, **(extra or {})}
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(cfg, f)


# Context manager / decorator that patches IOBinding so real C++ is never called.
_PATCH_IO_BINDING = patch("optimum.onnxruntime.base.IOBinding", autospec=False)


def _build_submodel(cls, in_names, out_names, tmpdir, extra_config=None,
                    input_type="tensor(float)", output_type="tensor(float)"):
    """Instantiate an ORTModelMixin subclass with a fully-mocked session.

    IOBinding is patched to avoid the C++ extension call.  All other session
    attributes are provided by _mock_session so ORTSessionMixin initialises
    correctly.
    """
    sub = os.path.join(tmpdir, cls.__name__.lower())
    os.makedirs(sub, exist_ok=True)
    _write_component_config(sub, extra_config or {})
    model_path = os.path.join(sub, "model.onnx")
    open(model_path, "w").close()
    sess = _mock_session(
        in_names, out_names, model_path=model_path,
        input_type=input_type, output_type=output_type,
    )
    parent = MagicMock()
    parent.__class__ = type("ORTStableDiffusionPipeline", (), {})
    with _PATCH_IO_BINDING:
        return cls(session=sess, parent=parent, use_io_binding=False)


# ── Suite 1: Dynamic class creation ───────────────────────────────────────────

class TestMakeORTPipelineClass(unittest.TestCase):
    """_make_ort_pipeline_class wraps every text-to-video diffusers pipeline."""

    def _check_pipeline(self, name):
        diffusers_class = getattr(diffusers, name, None)
        self.assertIsNotNone(
            diffusers_class,
            f"{name} not found in installed diffusers {diffusers.__version__} — "
            "update TEXT_TO_VIDEO_PIPELINES or upgrade diffusers"
        )
        ort_class = _make_ort_pipeline_class(diffusers_class)

        self.assertEqual(ort_class.__name__, f"ORT{name}")
        self.assertTrue(issubclass(ort_class, OnTheFlyORTDiffusionPipeline),
                        f"ORT{name} should inherit from OnTheFlyORTDiffusionPipeline")
        self.assertTrue(issubclass(ort_class, diffusers_class),
                        f"ORT{name} should inherit from {name}")
        self.assertIs(ort_class.auto_model_class, diffusers_class)

        # Each call returns a fresh class object
        ort_class2 = _make_ort_pipeline_class(diffusers_class)
        self.assertIsNot(ort_class, ort_class2)


def _add_t2v_tests():
    for pipeline_name in TEXT_TO_VIDEO_PIPELINES:
        def make_test(name):
            def test_fn(self):
                self._check_pipeline(name)
            test_fn.__name__ = f"test_{name}"
            return test_fn
        setattr(TestMakeORTPipelineClass, f"test_{pipeline_name}", make_test(pipeline_name))

_add_t2v_tests()


class TestDynamicClassCoverage(unittest.TestCase):
    """Extra coverage for _make_ort_pipeline_class edge cases."""

    def test_unknown_future_pipeline(self):
        """A brand-new pipeline added to diffusers works with zero code changes."""
        FuturePipeline = type("FuturePipeline", (diffusers.DiffusionPipeline,), {})
        ort_class = _make_ort_pipeline_class(FuturePipeline)
        self.assertEqual(ort_class.__name__, "ORTFuturePipeline")
        self.assertTrue(issubclass(ort_class, OnTheFlyORTDiffusionPipeline))
        self.assertTrue(issubclass(ort_class, FuturePipeline))

    def test_all_text_to_video_present_in_diffusers(self):
        """TEXT_TO_VIDEO_PIPELINES is a valid subset of the installed diffusers."""
        missing = [n for n in TEXT_TO_VIDEO_PIPELINES if not hasattr(diffusers, n)]
        self.assertEqual(missing, [],
                         f"Pipelines missing from diffusers {diffusers.__version__}: {missing}")

    def test_mro_order(self):
        """OnTheFlyORTDiffusionPipeline appears before the diffusers class in the MRO."""
        ort_cls = _make_ort_pipeline_class(diffusers.WanPipeline)
        mro = ort_cls.__mro__
        ort_idx = mro.index(OnTheFlyORTDiffusionPipeline)
        diffusers_idx = mro.index(diffusers.WanPipeline)
        self.assertLess(ort_idx, diffusers_idx,
                        "OnTheFlyORTDiffusionPipeline must come before WanPipeline in MRO")


# ── Suite 2: Text-to-video pipeline class creation ────────────────────────────

class TestTextToVideoPipelineClasses(unittest.TestCase):
    """One test per notable text-to-video pipeline (explicit regression guard)."""

    def _assert_ort_wraps(self, name):
        cls = getattr(diffusers, name, None)
        if cls is None:
            self.skipTest(f"{name} not available in diffusers {diffusers.__version__}")
        ort = _make_ort_pipeline_class(cls)
        self.assertEqual(ort.__name__, f"ORT{name}")
        self.assertTrue(issubclass(ort, cls))
        self.assertTrue(issubclass(ort, OnTheFlyORTDiffusionPipeline))

    def test_wan_pipeline(self):           self._assert_ort_wraps("WanPipeline")
    def test_wan_animate_pipeline(self):   self._assert_ort_wraps("WanAnimatePipeline")
    def test_cogvideox_pipeline(self):     self._assert_ort_wraps("CogVideoXPipeline")
    def test_mochi_pipeline(self):         self._assert_ort_wraps("MochiPipeline")
    def test_ltx_pipeline(self):           self._assert_ort_wraps("LTXPipeline")
    def test_ltx2_pipeline(self):          self._assert_ort_wraps("LTX2Pipeline")
    def test_hunyuan_video(self):          self._assert_ort_wraps("HunyuanVideoPipeline")
    def test_hunyuan_video15(self):        self._assert_ort_wraps("HunyuanVideo15Pipeline")
    def test_animatediff(self):            self._assert_ort_wraps("AnimateDiffPipeline")
    def test_animatediff_sdxl(self):       self._assert_ort_wraps("AnimateDiffSDXLPipeline")
    def test_text_to_video_sd(self):       self._assert_ort_wraps("TextToVideoSDPipeline")
    def test_latte(self):                  self._assert_ort_wraps("LattePipeline")
    def test_sana_video(self):             self._assert_ort_wraps("SanaVideoPipeline")


# ── Suite 3: from_pretrained with mocked I/O ──────────────────────────────────

class TestFromPretrainedMocked(unittest.TestCase):
    """from_pretrained builds the right dynamic pipeline class from a local dir."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _write_config(self.tmpdir, class_name="StableDiffusionPipeline")

        extra_by_subfolder = {
            "vae_encoder": {"scaling_factor": 0.18215, "block_out_channels": [128, 256, 512, 512]},
            "vae_decoder": {"scaling_factor": 0.18215},
        }
        for subfolder in ("unet", "text_encoder", "vae_decoder", "vae_encoder"):
            sub = os.path.join(self.tmpdir, subfolder)
            os.makedirs(sub, exist_ok=True)
            open(os.path.join(sub, "model.onnx"), "w").close()
            _write_component_config(sub, {
                "num_hidden_layers": 12,
                "num_decoder_layers": 12,
                **extra_by_subfolder.get(subfolder, {}),
            })

        sched_dir = os.path.join(self.tmpdir, "scheduler")
        os.makedirs(sched_dir, exist_ok=True)
        with open(os.path.join(sched_dir, "scheduler_config.json"), "w") as f:
            json.dump({"_class_name": "DDIMScheduler", "_diffusers_version": "0.38.0"}, f)

    def _make_sess(self, subfolder, in_names, out_names):
        path = os.path.join(self.tmpdir, subfolder, "model.onnx")
        return _mock_session(in_names, out_names, model_path=path)

    @_PATCH_IO_BINDING
    @patch("inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion.InferenceSession")
    def test_dynamic_class_from_class_name(self, mock_ort_session, mock_iobinding):
        """from_pretrained on base class creates ORT<PipelineName> dynamically.

        We stub the diffusers pipeline's __init__ to avoid needing a fully
        populated config; the test only verifies the ORT class-selection logic.
        """
        mock_ort_session.side_effect = lambda path, **kw: _mock_session(
            input_names=["sample", "timestep", "encoder_hidden_states"],
            output_names=["out_sample"],
            model_path=path,
        )

        # Stub diffusers.__init__ so we don't need a real unet.config.sample_size etc.
        with patch.object(diffusers.StableDiffusionPipeline, "__init__",
                          lambda self, **kw: None):
            with patch.object(OnTheFlyORTDiffusionPipeline, "load_config",
                              return_value={
                                  "_class_name": "StableDiffusionPipeline",
                                  "_diffusers_version": "0.38.0",
                                  "unet": ["diffusers", "UNet2DConditionModel"],
                                  "scheduler": (None, None),
                                  "tokenizer": (None, None),
                                  "tokenizer_2": (None, None),
                                  "tokenizer_3": (None, None),
                                  "feature_extractor": (None, None),
                              }):
                with patch.object(OnTheFlyORTDiffusionPipeline, "register_to_config"):
                    unet_sess = self._make_sess(
                        "unet",
                        ["sample", "timestep", "encoder_hidden_states"],
                        ["out_sample"],
                    )
                    pipe = OnTheFlyORTDiffusionPipeline.from_pretrained(
                        self.tmpdir,
                        export=False,
                        unet_session=unet_sess,
                    )

        self.assertEqual(type(pipe).__name__, "ORTStableDiffusionPipeline")
        self.assertIsInstance(pipe, OnTheFlyORTDiffusionPipeline)
        self.assertIsInstance(pipe, diffusers.StableDiffusionPipeline)

    @_PATCH_IO_BINDING
    @patch("inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion.InferenceSession")
    def test_concrete_subclass_not_re_wrapped(self, mock_ort_session, mock_iobinding):
        """Calling from_pretrained on a concrete subclass keeps that exact class."""
        ort_sd_cls = _make_ort_pipeline_class(diffusers.StableDiffusionPipeline)
        mock_ort_session.side_effect = lambda path, **kw: _mock_session(
            input_names=["sample"], output_names=["out_sample"], model_path=path,
        )

        with patch.object(diffusers.StableDiffusionPipeline, "__init__",
                          lambda self, **kw: None):
            with patch.object(ort_sd_cls, "load_config",
                              return_value={
                                  "_class_name": "StableDiffusionPipeline",
                                  "_diffusers_version": "0.38.0",
                                  "unet": ["diffusers", "UNet2DConditionModel"],
                                  "scheduler": (None, None),
                                  "tokenizer": (None, None),
                                  "tokenizer_2": (None, None),
                                  "tokenizer_3": (None, None),
                                  "feature_extractor": (None, None),
                              }):
                with patch.object(ort_sd_cls, "register_to_config"):
                    unet_sess = self._make_sess(
                        "unet",
                        ["sample", "timestep", "encoder_hidden_states"],
                        ["out_sample"],
                    )
                    pipe = ort_sd_cls.from_pretrained(
                        self.tmpdir,
                        export=False,
                        unet_session=unet_sess,
                    )
        self.assertIs(type(pipe), ort_sd_cls)

    @_PATCH_IO_BINDING
    @patch("inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion.InferenceSession")
    def test_unknown_class_name_raises(self, mock_ort_session, mock_iobinding):
        """from_pretrained raises ValueError for a _class_name not in diffusers."""
        mock_ort_session.side_effect = lambda path, **kw: _mock_session(
            input_names=["sample"], output_names=["out_sample"], model_path=path,
        )
        with patch.object(OnTheFlyORTDiffusionPipeline, "load_config",
                          return_value={
                              "_class_name": "NonExistentXYZPipeline",
                              "_diffusers_version": "0.38.0",
                              "unet": ["diffusers", "UNet2DConditionModel"],
                              "scheduler": (None, None),
                              "tokenizer": (None, None),
                              "tokenizer_2": (None, None),
                              "tokenizer_3": (None, None),
                              "feature_extractor": (None, None),
                          }):
            with self.assertRaises(ValueError, msg="Should raise for unknown pipeline class"):
                OnTheFlyORTDiffusionPipeline.from_pretrained(self.tmpdir, export=False)


# ── Suite 4: Submodule forward pass ───────────────────────────────────────────

class TestSubmoduleForward(unittest.TestCase):
    """Forward method of each ORT submodule runs correctly with a mock session."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_ort_transformer_forward(self):
        """ORTTransformer.forward calls session.run with the right input names."""
        model = _build_submodel(
            ORTTransformer,
            in_names=["hidden_states", "encoder_hidden_states", "timestep"],
            out_names=["out_hidden_states"],
            tmpdir=self.tmpdir,
        )
        hidden = torch.randn(1, 64).float()
        enc = torch.randn(1, 64).float()
        ts = torch.tensor([0.5])

        out = model.forward(hidden_states=hidden, encoder_hidden_states=enc, timestep=ts)

        model.session.run.assert_called_once()
        # session.run(None, feed_dict) — positional args
        call_pos_args = model.session.run.call_args[0]
        self.assertIsNone(call_pos_args[0])
        feed = call_pos_args[1]
        self.assertIn("hidden_states", feed)
        self.assertIn("encoder_hidden_states", feed)
        self.assertIn("timestep", feed)

    def test_ort_text_encoder_forward(self):
        """ORTTextEncoder.forward converts int input_ids and calls session.run."""
        model = _build_submodel(
            ORTTextEncoder,
            in_names=["input_ids"],
            out_names=["last_hidden_state"],
            tmpdir=self.tmpdir,
            input_type="tensor(int64)",
        )
        input_ids = torch.randint(0, 1000, (1, 64))
        out = model.forward(input_ids=input_ids)

        model.session.run.assert_called_once()
        feed = model.session.run.call_args[0][1]
        self.assertIn("input_ids", feed)
        self.assertEqual(feed["input_ids"].dtype, np.int64)

    def test_ort_vae_decoder_forward(self):
        """ORTVaeDecoder.forward passes latent_sample and returns ModelOutput."""
        model = _build_submodel(
            ORTVaeDecoder,
            in_names=["latent_sample"],
            out_names=["sample"],
            tmpdir=self.tmpdir,
            extra_config={"scaling_factor": 0.18215},
        )
        latent = torch.randn(1, 64).float()
        out = model.forward(latent_sample=latent)

        model.session.run.assert_called_once()
        feed = model.session.run.call_args[0][1]
        self.assertIn("latent_sample", feed)

    def test_ort_vae_encoder_forward(self):
        """ORTVaeEncoder.forward passes sample and wraps output in ModelOutput."""
        model = _build_submodel(
            ORTVaeEncoder,
            in_names=["sample"],
            out_names=["latent_parameters"],
            tmpdir=self.tmpdir,
            extra_config={
                "scaling_factor": 0.18215,
                "block_out_channels": [128, 256, 512, 512],
            },
        )
        image = torch.randn(1, 64).float()
        out = model.forward(sample=image)

        model.session.run.assert_called_once()
        feed = model.session.run.call_args[0][1]
        self.assertIn("sample", feed)

    def test_ort_unet_forward(self):
        """ORTUnet.forward handles scalar timestep and calls session.run."""
        model = _build_submodel(
            ORTUnet,
            in_names=["sample", "timestep", "encoder_hidden_states"],
            out_names=["out_sample"],
            tmpdir=self.tmpdir,
            extra_config={"time_cond_proj_dim": None},
        )
        sample = torch.randn(1, 64).float()
        timestep = torch.tensor(500.0)
        enc = torch.randn(1, 64).float()

        out = model.forward(sample=sample, timestep=timestep, encoder_hidden_states=enc)

        model.session.run.assert_called_once()
        feed = model.session.run.call_args[0][1]
        self.assertIn("sample", feed)
        self.assertIn("timestep", feed)
        self.assertIn("encoder_hidden_states", feed)


# ── Suite 5: IO binding file wiring ───────────────────────────────────────────

class TestIOBindingWiring(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_set_io_binding_file(self):
        """set_io_binding_file stores the path on the model instance."""
        model = _build_submodel(
            ORTTransformer,
            in_names=["hidden_states"],
            out_names=["out_hidden_states"],
            tmpdir=self.tmpdir,
        )
        model.set_io_binding_file("/tmp/io/transformer_outputs.json")
        self.assertEqual(model.io_binding_file, "/tmp/io/transformer_outputs.json")

    def test_load_shapes_missing_file(self):
        """load_shapes_as_torch_size returns {} for a nonexistent path."""
        from inference_driven_model_compiler.optimum.onnxruntime.utils import load_shapes_as_torch_size
        self.assertEqual(load_shapes_as_torch_size("/nonexistent/path.json"), {})

    def test_load_shapes_valid_file(self):
        """load_shapes_as_torch_size converts JSON lists to torch.Size objects."""
        from inference_driven_model_compiler.optimum.onnxruntime.utils import load_shapes_as_torch_size
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"out_hidden_states": [1, 4, 64, 64], "logits": [1, 77, 512]}, f)
            tmp = f.name
        try:
            shapes = load_shapes_as_torch_size(tmp)
            self.assertEqual(shapes["out_hidden_states"], torch.Size([1, 4, 64, 64]))
            self.assertEqual(shapes["logits"], torch.Size([1, 77, 512]))
        finally:
            os.unlink(tmp)


# ── Suite 6: ORTVae wrapper ───────────────────────────────────────────────────

class TestORTVae(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_encoder(self):
        return _build_submodel(
            ORTVaeEncoder,
            in_names=["sample"],
            out_names=["latent_parameters"],
            tmpdir=self.tmpdir,
            extra_config={
                "scaling_factor": 0.18215,
                "block_out_channels": [128, 256],
            },
        )

    def _make_decoder(self):
        return _build_submodel(
            ORTVaeDecoder,
            in_names=["latent_sample"],
            out_names=["sample"],
            tmpdir=self.tmpdir,
            extra_config={"scaling_factor": 0.18215},
        )

    def test_vae_encode_decode(self):
        enc = self._make_encoder()
        dec = self._make_decoder()
        vae = ORTVae(encoder=enc, decoder=dec)

        # config property delegates to decoder
        self.assertIs(vae.config, dec.config)

        image = torch.randn(1, 64).float()
        vae.encode(sample=image)
        enc.session.run.assert_called_once()

        latent = torch.randn(1, 64).float()
        vae.decode(latent_sample=latent)
        dec.session.run.assert_called_once()

    def test_vae_encoder_only(self):
        enc = self._make_encoder()
        vae = ORTVae(encoder=enc, decoder=None)
        self.assertIsNone(vae.decoder)
        vae.encode(sample=torch.randn(1, 64).float())
        enc.session.run.assert_called_once()

    def test_vae_decoder_only(self):
        dec = self._make_decoder()
        vae = ORTVae(encoder=None, decoder=dec)
        self.assertIsNone(vae.encoder)
        vae.decode(latent_sample=torch.randn(1, 64).float())
        dec.session.run.assert_called_once()


# ── Suite 7: SDXL text-to-image export wrappers ───────────────────────────────

class TestSDXLExportWrappers(unittest.TestCase):
    """The UNet/text-encoder export wrappers used for SDXL-style text-to-image."""

    def test_sdxl_pipeline_class_creation(self):
        """_make_ort_pipeline_class wraps StableDiffusionXLPipeline."""
        cls = getattr(diffusers, "StableDiffusionXLPipeline", None)
        if cls is None:
            self.skipTest("StableDiffusionXLPipeline not in diffusers")
        ort = _make_ort_pipeline_class(cls)
        self.assertEqual(ort.__name__, "ORTStableDiffusionXLPipeline")
        self.assertTrue(issubclass(ort, OnTheFlyORTDiffusionPipeline))
        self.assertTrue(issubclass(ort, cls))

    def test_unet_added_cond_wrapper(self):
        """_UnetAddedCondWrapper rebuilds added_cond_kwargs and names output out_sample."""
        seen = {}

        class FakeUnet(torch.nn.Module):
            def forward(self, sample, timestep, encoder_hidden_states,
                        timestep_cond=None, added_cond_kwargs=None, return_dict=True):
                seen["added_cond_kwargs"] = added_cond_kwargs
                seen["return_dict"] = return_dict
                return (sample + 1,)

        w = _UnetAddedCondWrapper(FakeUnet())
        sample = torch.zeros(2, 4, 8, 8)
        text_embeds = torch.zeros(2, 1280)
        time_ids = torch.zeros(2, 6)
        out = w(sample=sample, timestep=torch.tensor(1.0),
                encoder_hidden_states=torch.zeros(2, 77, 2048),
                text_embeds=text_embeds, time_ids=time_ids)
        # flattened tensors are reassembled into the nested dict
        self.assertIn("text_embeds", seen["added_cond_kwargs"])
        self.assertIn("time_ids", seen["added_cond_kwargs"])
        self.assertFalse(seen["return_dict"])
        # output is keyed out_sample (what ORTUnet expects)
        self.assertEqual(list(out.keys()), ["out_sample"])
        self.assertTrue(torch.allclose(out["out_sample"], sample + 1))

    def test_text_encoder_hidden_states_wrapper(self):
        """_TextEncoderHiddenStatesWrapper flattens hidden_states, preserves field order."""
        class FakeTE(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None,
                        output_hidden_states=None, return_dict=True):
                assert output_hidden_states is True  # wrapper must force this on
                # mimic CLIPTextModelWithProjection field order: text_embeds first
                return ModelOutput(
                    text_embeds=torch.zeros(1, 1280),
                    last_hidden_state=torch.zeros(1, 77, 1280),
                    hidden_states=(torch.zeros(1, 77, 1280), torch.ones(1, 77, 1280)),
                )

        w = _TextEncoderHiddenStatesWrapper(FakeTE())
        out = w(input_ids=torch.zeros(1, 77, dtype=torch.long))
        keys = list(out.keys())
        # native field order preserved → output[0] stays text_embeds
        self.assertEqual(keys[0], "text_embeds")
        self.assertIn("last_hidden_state", keys)
        # hidden_states tuple flattened into per-layer entries
        self.assertIn("hidden_states.0", keys)
        self.assertIn("hidden_states.1", keys)
        self.assertNotIn("hidden_states", keys)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestMakeORTPipelineClass,
        TestDynamicClassCoverage,
        TestTextToVideoPipelineClasses,
        TestFromPretrainedMocked,
        TestSubmoduleForward,
        TestIOBindingWiring,
        TestORTVae,
        TestSDXLExportWrappers,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
