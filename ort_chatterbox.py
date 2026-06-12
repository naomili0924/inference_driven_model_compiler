"""ORTChatterboxPipeline — run ResembleAI/chatterbox-turbo with ONNX Runtime.

Mirrors how the ORT diffusion pipeline works: it loads the real
``ChatterboxTurboTTS`` and runs Chatterbox's OWN generation logic unchanged
(the T3 autoregressive loop, the S3Gen CFM Euler solver, the HiFiGAN vocoder,
conditioning, the stochastic NSF source generator) — but swaps the four heavy
*leaf* compute calls for ORT sessions over the ONNX graphs produced by
``chatterbox_export.py``:

    tts.t3.tfmr                          -> t3_backbone/model.onnx   (GPT2, with-past)
    tts.s3gen.flow.decoder.estimator     -> s3gen_estimator/model.onnx (CFM step)
    tts.s3gen.mel2wav.decode             -> s3gen_hift/model.onnx    (mel->wav)
    tts.ve                               -> ve/model.onnx            (speaker embed)

The loops that drive these stay in Python and call the ORT sessions N times —
exactly like the diffusers scheduler calling a single-step UNet.

Usage:

    HF_HOME=/dev/shm/hf PYTHONPATH=/workspace/inference_driven_model_compiler \
    /opt/cbx-venv/bin/python -c "
    from ort_chatterbox import ORTChatterboxPipeline
    pipe = ORTChatterboxPipeline('/dev/shm/cbx_onnx', device='cuda')
    wav = pipe.generate('Hello from ONNX Runtime.')
    import torchaudio; torchaudio.save('out.wav', wav, pipe.sr)"
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float32).contiguous().numpy()


class _OrtSession:
    """Thin ORT session wrapper that runs on numpy and remembers I/O names."""

    def __init__(self, path: str, providers):
        self.sess = ort.InferenceSession(path, providers=providers)
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feeds: dict) -> dict:
        outs = self.sess.run(None, feeds)
        return dict(zip(self.out_names, outs))


class _ORTBackbone(torch.nn.Module):
    """ORT stand-in for t3.tfmr (GPT2 decoder-with-past on inputs_embeds).

    An nn.Module so it can replace the registered ``tfmr`` submodule. Accepts the
    same call signature t3.inference_turbo uses (``inputs_embeds``,
    ``past_key_values`` as a DynamicCache or None, ``use_cache``) and returns a
    ``BaseModelOutputWithPast`` so that ``out[0]`` -> last_hidden_state and
    ``out.past_key_values`` -> DynamicCache.
    """

    def __init__(self, sess: _OrtSession, n_layer, n_head, head_dim):
        super().__init__()
        self.sess = sess
        self.L, self.H, self.D = n_layer, n_head, head_dim

    def forward(self, inputs_embeds, past_key_values=None, use_cache=True, **kw):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        dev = inputs_embeds.device
        B = inputs_embeds.shape[0]
        feeds = {"inputs_embeds": _np(inputs_embeds)}
        if past_key_values is None or len(past_key_values) == 0:
            empty = np.zeros((B, self.H, 0, self.D), dtype=np.float32)
            for i in range(self.L):
                feeds[f"past_key_values.{i}.key"] = empty
                feeds[f"past_key_values.{i}.value"] = empty
        else:
            for i, layer in enumerate(past_key_values.layers):
                feeds[f"past_key_values.{i}.key"] = _np(layer.keys)
                feeds[f"past_key_values.{i}.value"] = _np(layer.values)

        out = self.sess.run(feeds)
        hidden = torch.from_numpy(out["last_hidden_state"]).to(dev)
        cache = DynamicCache()
        for i in range(self.L):
            k = torch.from_numpy(out[f"present.{i}.key"]).to(dev)
            v = torch.from_numpy(out[f"present.{i}.value"]).to(dev)
            cache.update(k, v, i)
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache)


_LEAF_DIRS = ("ve", "s3gen_estimator", "s3gen_hift", "t3_backbone")


class ORTChatterboxPipeline:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, export: bool | None = None,
                        output: str | None = None, export_device: str = "cpu",
                        device: str = "cuda", provider: str | None = None,
                        token: str | None = None, cache_dir: str | None = None,
                        revision: str | None = None, **kwargs):
        """Load a chatterbox-turbo ONNX export and build the runtime pipeline.

        Two modes, mirroring the other ORT pipelines' ``from_pretrained``:

        * **load** (``export=False``) — ``model_name_or_path`` is a local dir or a
          Hub repo id holding the four leaf sub-folders (``ve/``,
          ``s3gen_estimator/``, ``s3gen_hift/``, ``t3_backbone/`` with
          ``model.onnx``); they are downloaded/loaded as-is.
        * **export** (``export=True``) — export the four leaves on the fly from the
          torch ``chatterbox-turbo`` checkpoint into ``output`` (a temp dir if
          omitted), then load. This is the inference-driven export path; unlike the
          diffusion/transformer pipelines it takes no ``inference_kwargs`` /
          ``module_fixed_axis_fields`` — each leaf's tracing is bespoke and lives
          in ``chatterbox_export.py``, so there is nothing generic to parameterize.

        ``export=None`` (default) auto-detects: load when the four ONNX graphs are
        already present locally, otherwise export.
        """
        p = Path(model_name_or_path)
        has_local_onnx = p.is_dir() and all((p / d / "model.onnx").exists() for d in _LEAF_DIRS)

        if export is None:
            export = not has_local_onnx

        if export:
            # The export driver needs the *shadow* optimum (this repo dir on
            # sys.path ahead of site-packages) so `import optimum` resolves to it.
            import sys
            repo_dir = str(Path(__file__).resolve().parent)
            if sys.path[:1] != [repo_dir]:
                sys.path.insert(0, repo_dir)
            from chatterbox_export import export_all
            if output is None:
                import tempfile
                output = tempfile.mkdtemp(prefix="cbx_onnx_")
            results = export_all(output, device=export_device)
            failed = [k for k, ok in results.items() if not ok]
            if failed:
                raise RuntimeError(f"chatterbox export failed for: {failed}")
            local_dir = output
        elif has_local_onnx:
            local_dir = str(p)
        else:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                model_name_or_path, repo_type="model", token=token,
                cache_dir=cache_dir, revision=revision,
                allow_patterns=[f"{d}/*" for d in _LEAF_DIRS],
            )
        return cls(local_dir, device=device, provider=provider)

    def __init__(self, onnx_dir: str, device: str = "cuda", provider: str | None = None):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        onnx_dir = Path(onnx_dir)
        if provider is None:
            provider = ("CUDAExecutionProvider"
                        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers()
                        else "CPUExecutionProvider")
        providers = [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]

        self.tts = ChatterboxTurboTTS.from_pretrained(device)
        self.sr = self.tts.sr
        self.device = device

        # capture GPT2 dims before replacing the backbone
        cfg = self.tts.t3.tfmr.config
        L, H = cfg.n_layer, cfg.n_head
        D = cfg.n_embd // cfg.n_head

        sess = lambda n: _OrtSession(str(onnx_dir / n / "model.onnx"), providers)
        self._s_t3 = sess("t3_backbone")
        self._s_est = sess("s3gen_estimator")
        self._s_hift = sess("s3gen_hift")
        self._s_ve = sess("ve")

        self._swap_leaves(L, H, D)

    def _swap_leaves(self, L, H, D):
        tts = self.tts

        # 1) T3 GPT2 backbone -> ORT (replace the whole module; frees torch weights)
        tts.t3.tfmr = _ORTBackbone(self._s_t3, L, H, D)

        # 2) S3Gen CFM estimator: called as self.estimator.forward(x=..,...)
        est_sess = self._s_est
        def est_forward(x, mask, mu, t, spks=None, cond=None, r=None):
            feeds = {"x": _np(x), "mask": _np(mask), "mu": _np(mu), "t": _np(t),
                     "spks": _np(spks), "cond": _np(cond), "r": _np(r)}
            out = est_sess.run(feeds)["d_mel"]
            return torch.from_numpy(out).to(x.device)
        tts.s3gen.flow.decoder.estimator.forward = est_forward

        # 3) HiFiGAN: called as self.decode(x=mel, s=source); source-gen stays torch
        hift_sess = self._s_hift
        def decode(x, s=torch.zeros(1, 1, 0)):
            feeds = {"speech_feat": _np(x), "source": _np(s)}
            out = hift_sess.run(feeds)["wav"]
            return torch.from_numpy(out).to(x.device)
        tts.s3gen.mel2wav.decode = decode

        # 4) Voice encoder: called as self.forward(mels) inside embeds_from_wavs
        ve_sess = self._s_ve
        def ve_forward(mels):
            out = ve_sess.run({"mels": _np(mels)})["speaker_embed"]
            return torch.from_numpy(out).to(mels.device)
        tts.ve.forward = ve_forward

    def prepare_conditionals(self, *a, **kw):
        return self.tts.prepare_conditionals(*a, **kw)

    def generate(self, text, **kw):
        return self.tts.generate(text, **kw)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/dev/shm/cbx_onnx")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text", default="Hello from ONNX Runtime.")
    ap.add_argument("--out", default="/dev/shm/ort_chatterbox_out.wav")
    args = ap.parse_args()

    pipe = ORTChatterboxPipeline(args.onnx, device=args.device)
    wav = pipe.generate(args.text)
    print("generated wav:", tuple(wav.shape), "sr", pipe.sr)
    try:
        import torchaudio
        torchaudio.save(args.out, wav, pipe.sr)
        print("saved", args.out)
    except Exception as e:
        print("save skipped:", e)
