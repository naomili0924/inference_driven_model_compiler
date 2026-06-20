"""Inference-driven ONNX export driver for ResembleAI/chatterbox-turbo.

Chatterbox is neither a ``transformers`` nor a ``diffusers`` checkpoint, so it
cannot go through ``optimum-cli export onnx`` / ``main_export`` (which load via
``AutoConfig`` / ``TasksManager``). Instead this driver loads the model with the
``chatterbox`` package, then feeds each *leaf* ``nn.Module`` (the per-call
networks underneath Chatterbox's autoregressive / ODE loops) through the
pipeline's generic primitives:

    coherent shape trials  ->  _compute_dynamic_axes()  (empirical dynamic axes)
    DummyOnnxConfig        ->  export_models() + validate_models_outputs()

No edits to the shared pipeline are required: the leaf modules are plain
``nn.Module``s with a single, stateless ``forward``. The loops that drive them
(T3 AR sampling, the S3Gen CFM Euler solver, HiFiGAN) live in the runtime
``OnTheFlyORTChatterboxPipeline`` instead — mirroring how the diffusers scheduler loop
calls an exported single-step UNet N times.

Run with the chatterbox venv + the repo on PYTHONPATH:

    HF_HOME=/dev/shm/hf \
    PYTHONPATH=/workspace/inference_driven_model_compiler \
    /opt/cbx-venv/bin/python -m chatterbox_export --modules ve,s3gen_estimator \
        --output /dev/shm/cbx_onnx
"""
from __future__ import annotations

import argparse
import math
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from optimum.exporters.onnx.utils import _compute_dynamic_axes
from optimum.exporters.onnx.model_configs import DummyOnnxConfig
from optimum.exporters.onnx import export_models, validate_models_outputs


# ---------------------------------------------------------------------------
# Leaf module specs
# ---------------------------------------------------------------------------
# Each spec describes one exportable leaf:
#   get       : tts -> nn.Module to export (its .forward is the graph)
#   make_inputs(size) -> dict[name -> tensor]  (ORDERED to match forward sig)
#   sizes     : list of coherent trial sizes for empirical dynamic-axis detection
#   out_names : names for the forward output tensor(s)
#   fixed     : input names whose *real value* must be replayed verbatim at
#               export/validation (e.g. value-guarded inputs); others are
#               regenerated randomly from their shape.
# ---------------------------------------------------------------------------

def _ve_inputs(size, device, module):
    # VoiceEncoder.forward(mels: (B, 160, 40)); guarded to require mels in [0,1].
    B = size
    return {"mels": torch.rand(B, 160, 40, device=device)}


def _patch_hift_stft(module, device):
    """Replace HiFTGenerator._stft / _istft (torch.stft/istft, complex — not ONNX
    exportable) with conv/matmul equivalents using the module's own window.
    Verified to match torch.stft (~7e-6) and torch.istft (~2e-7)."""
    N = module.istft_params["n_fft"]
    H = module.istft_params["hop_len"]
    win = module.stft_window.to(device).float()
    Fb = N // 2 + 1
    n = torch.arange(N, device=device).float()
    k = torch.arange(Fb, device=device).float()
    ang = 2 * math.pi * k[:, None] * n[None, :] / N         # (F, N)
    w_real = (torch.cos(ang) * win[None, :])[:, None, :]    # (F,1,N)
    w_imag = (-torch.sin(ang) * win[None, :])[:, None, :]
    c = torch.ones(Fb, device=device); c[1:Fb - 1] = 2.0    # onesided->full
    Br = (torch.cos(ang).T * c[None, :]) / N                # (N, F)
    Bi = (-torch.sin(ang).T * c[None, :]) / N
    Bi[:, 0] = 0.0; Bi[:, -1] = 0.0                         # irfft ignores im(k=0,N/2)
    eye = torch.eye(N, device=device)[:, None, :]           # (N,1,N) overlap-add

    def _stft(self, x):
        xp = F.pad(x[:, None, :], (N // 2, N // 2), mode="reflect")
        return F.conv1d(xp, w_real, stride=H), F.conv1d(xp, w_imag, stride=H)

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        frame = (torch.einsum("nf,bft->bnt", Br, real)
                 + torch.einsum("nf,bft->bnt", Bi, imag)) * win[None, :, None]
        y = F.conv_transpose1d(frame, eye, stride=H)
        T = real.shape[-1]
        wsq = (win ** 2)[None, :, None].expand(1, N, T)
        env = F.conv_transpose1d(wsq, eye, stride=H)
        return (y / env.clamp_min(1e-8))[:, 0, N // 2:-(N // 2)]

    module._stft = types.MethodType(_stft, module)
    module._istft = types.MethodType(_istft, module)
    return module


def _hift_forward(self, speech_feat, source):
    # Export HiFTGenerator.decode(x=mel, s=source) -> waveform. The stochastic
    # NSF source generator (f0_predictor + m_source; uses RNG) stays in Python
    # and feeds `source` in; the two embedded STFTs are conv-replaced.
    return self.decode(x=speech_feat, s=source)


def _hift_inputs(size, device, module):
    # mel (1,80,Tm) + a coherent source (1,1,L) built by the real source path.
    speech_feat = torch.randn(1, 80, size, device=device)
    with torch.no_grad():
        f0 = module.f0_predictor(speech_feat)
        s = module.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = module.m_source(s)
        s = s.transpose(1, 2)
    return {"speech_feat": speech_feat, "source": s}


def _estimator_inputs(size, device, module):
    # ConditionalDecoder.forward(x, mask, mu, t, spks, cond, r) ; T is dynamic.
    T = size
    return {
        "x":    torch.randn(1, 80, T, device=device),
        "mask": torch.ones(1, 1, T, device=device),
        "mu":   torch.randn(1, 80, T, device=device),
        "t":    torch.rand(1, device=device),
        "spks": torch.randn(1, 80, device=device),
        "cond": torch.randn(1, 80, T, device=device),
        "r":    torch.rand(1, device=device),
    }


LEAVES = {
    "ve": dict(
        get=lambda tts: tts.ve,
        make_inputs=_ve_inputs,
        sizes=[7, 4, 11],          # batch = number of partial utterances
        out_names=["speaker_embed"],
        fixed=["mels"],            # value-guarded to [0,1] -> replay verbatim
    ),
    "s3gen_estimator": dict(
        get=lambda tts: tts.s3gen.flow.decoder.estimator,
        make_inputs=_estimator_inputs,
        sizes=[432, 300, 200],     # mel time dim (2 x frames); dynamic
        out_names=["d_mel"],
        fixed=[],
    ),
    "s3gen_hift": dict(
        get=lambda tts: tts.s3gen.mel2wav,
        prepare=_patch_hift_stft,     # conv-replace the two torch.stft/istft
        bind_forward=_hift_forward,   # expose decode(mel, source)->wav
        make_inputs=_hift_inputs,
        sizes=[132, 100, 200],        # mel frames; dynamic -> dynamic samples
        out_names=["wav"],
        fixed=[],
    ),
}


def _flatten_outputs(out, names):
    """Map a forward output (tensor or tuple) to {name: shape_tuple}."""
    if torch.is_tensor(out):
        outs = [out]
    elif isinstance(out, (tuple, list)):
        outs = [o for o in out if torch.is_tensor(o)]
    else:
        raise TypeError(f"Unhandled output type {type(out)}")
    if len(outs) != len(names):
        raise ValueError(f"Expected {len(names)} outputs, got {len(outs)}")
    return {n: tuple(o.shape) for n, o in zip(names, outs)}


def export_leaf(name: str, tts, output_dir: Path, device: str = "cpu",
                opset: int = 17, atol: float = 1e-3) -> bool:
    spec = LEAVES[name]
    module = spec["get"](tts).to(device).eval()
    if spec.get("prepare") is not None:
        # Custom pre-export surgery (e.g. conv-replace non-exportable STFTs).
        module = spec["prepare"](module, device)
    if spec.get("bind_forward") is not None:
        # Swap in a clean single-purpose forward (e.g. decode()->wav), the
        # same technique the diffusers branch uses for the VAE decoder.
        module.forward = types.MethodType(spec["bind_forward"], module)
    make_inputs = spec["make_inputs"]
    out_names = spec["out_names"]
    fixed = set(spec["fixed"])

    print(f"\n{'='*70}\n[{name}] {type(module).__module__}.{type(module).__name__}")

    # --- coherent multi-trial tracing -> empirical dynamic axes ---
    all_in_shapes, all_out_shapes = [], []
    trial0_inputs = None
    for i, size in enumerate(spec["sizes"]):
        kw = make_inputs(size, device, module)
        if i == 0:
            trial0_inputs = kw
        with torch.no_grad():
            out = module(**kw)
        all_in_shapes.append({k: tuple(v.shape) for k, v in kw.items()})
        all_out_shapes.append(_flatten_outputs(out, out_names))
    dyn = _compute_dynamic_axes(all_in_shapes, all_out_shapes)
    print(f"[{name}] dynamic_axes: {dyn}")

    # --- build DummyOnnxConfig from trial 0 ---
    # model_inputs: shape tuples, except 'fixed' inputs stored as real tensors
    # (replayed verbatim, bypassing value guards / random regeneration).
    model_inputs = {}
    model_input_dtypes = {}
    for k, v in trial0_inputs.items():
        model_inputs[k] = v.detach().to("cpu") if k in fixed else tuple(v.shape)
        model_input_dtypes[k] = v.dtype
    with torch.no_grad():
        out0 = module(**trial0_inputs)
    model_outputs = _flatten_outputs(out0, out_names)

    # Bare nn.Modules have no .config; stock optimum export_pytorch does
    # `model.config.return_dict = True`, so attach a minimal mutable namespace.
    cfg_ns = SimpleNamespace(
        model_type=f"chatterbox_{name}", return_dict=True,
        is_encoder_decoder=False, use_cache=False,
    )
    if not hasattr(module, "config"):
        module.config = cfg_ns

    cfg = DummyOnnxConfig(
        config=cfg_ns,
        task="backbone",
        model_inputs=model_inputs,
        model_outputs=model_outputs,
        dynamic_axes=dyn,
        model_input_dtypes=model_input_dtypes,
        float_dtype="fp32",
    )
    return _export_and_validate(name, module, cfg, output_dir, device, opset, atol)


def _export_and_validate(name, module, cfg, output_dir, device, opset, atol,
                         disable_dynamic_axes_fix=True, dtype="fp32"):
    print(f"[{name}] onnx inputs:  {list(cfg.inputs)}")
    print(f"[{name}] onnx outputs: {list(cfg.outputs)}")
    subpath = f"{name}/model.onnx"
    models_and_onnx_configs = {name: (module, cfg)}
    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=output_dir,
        opset=opset,
        output_names=[subpath],
        device=device,
        dtype=dtype,
        disable_dynamic_axes_fix=disable_dynamic_axes_fix,
        do_constant_folding=True,
    )
    print(f"[{name}] exported -> {output_dir / subpath}")
    validate_models_outputs(
        models_and_onnx_configs=models_and_onnx_configs,
        onnx_named_outputs=onnx_outputs,
        atol=atol,
        output_dir=output_dir,
        onnx_files_subpaths=[subpath],
        input_shapes=None,
        device=device,
        use_subprocess=False,
        model_kwargs=None,
    )
    print(f"[{name}] ✅ validation passed (atol={atol})")
    return True


# ---------------------------------------------------------------------------
# t3_backbone: GPT2 decoder-with-past on inputs_embeds (special case)
# ---------------------------------------------------------------------------

class _T3BackboneWrapper(torch.nn.Module):
    """Wrap t3.tfmr (GPT2Model, wte deleted -> inputs_embeds only) so it (a) takes
    past_key_values as a legacy tuple (what DummyOnnxConfig.generate_dummy_inputs
    produces) and converts it to a transformers-5.x DynamicCache, and (b) returns
    a FLAT tuple (last_hidden_state, present_0_key, present_0_value, ...) so the
    ONNX graph has plain tensor I/O (no Cache objects in the pytree)."""

    def __init__(self, tfmr):
        super().__init__()
        self.tfmr = tfmr
        self.n_layer = tfmr.config.n_layer
        self.config = SimpleNamespace(
            model_type="chatterbox_t3_backbone", return_dict=True,
            is_encoder_decoder=False, use_cache=True,
        )

    def forward(self, inputs_embeds, past_key_values=None, use_cache=True):
        from transformers.cache_utils import DynamicCache
        if isinstance(past_key_values, (tuple, list)):
            dc = DynamicCache()
            for i, (k, v) in enumerate(past_key_values):
                dc.update(k, v, i)
            past_key_values = dc
        out = self.tfmr(inputs_embeds=inputs_embeds,
                        past_key_values=past_key_values, use_cache=True)
        flat = [out.last_hidden_state]
        for layer in out.past_key_values.layers:
            flat.append(layer.keys)
            flat.append(layer.values)
        return tuple(flat)


def _convert_onnx_to_fp16(path):
    """Convert an exported fp32 ONNX to fp16 weights in place.

    Uses onnxconverter_common with ``keep_io_types=True`` (graph I/O stays fp32,
    so the runtime needs no changes) and keeps ``LayerNormalization`` in fp32 —
    casting GPT2's LayerNorm to fp16 yields mixed-precision type errors and hurts
    numerical stability. Halves the big matmul weights; LayerNorm weights are tiny.
    """
    import onnx
    from onnxruntime.transformers.onnx_model import OnnxModel
    before = Path(path).stat().st_size
    # ORT's transformer-aware fp16 converter (not onnxconverter_common, which
    # mis-places boundary casts on this 49-I/O with-past graph -> mixed-type Adds).
    # keep_io_types=True leaves graph I/O fp32 so the runtime needs no changes.
    om = OnnxModel(onnx.load(str(path)))
    om.convert_float_to_float16(keep_io_types=True)
    om.save_model_to_file(str(path))
    after = Path(path).stat().st_size
    print(f"    fp16 conversion: {before/1e6:.0f} MB -> {after/1e6:.0f} MB")
    return True


def export_t3_backbone(tts, output_dir: Path, device="cpu", opset=17, atol=1e-3,
                       dtype="fp32"):
    name = "t3_backbone"
    use_fp16 = dtype in ("fp16", "float16")
    # Always trace/export in fp32 (GPT2 fp16 tracing produces mixed-precision
    # LayerNorm); fp16 is applied as a post-export ONNX conversion below.
    tfmr = tts.t3.tfmr.to(device).eval()
    L = tfmr.config.n_layer
    H = tfmr.config.n_head
    D = tfmr.config.n_embd // tfmr.config.n_head
    E = tfmr.config.n_embd
    wrapper = _T3BackboneWrapper(tfmr).to(device).eval()
    print(f"\n{'='*70}\n[{name}] GPT2Model decoder-with-past  L={L} H={H} D={D} E={E} dtype={dtype}")

    def mk_inputs(q, P):
        emb = torch.randn(1, q, E, device=device)
        past = tuple((torch.randn(1, H, P, D, device=device),
                      torch.randn(1, H, P, D, device=device)) for _ in range(L))
        return emb, past

    def in_shapes(emb, past):
        d = {"inputs_embeds": tuple(emb.shape)}
        for i, (k, v) in enumerate(past):
            d[f"past_key_values.{i}.key"] = tuple(k.shape)
            d[f"past_key_values.{i}.value"] = tuple(v.shape)
        return d

    def out_shapes(flat):
        d = {"last_hidden_state": tuple(flat[0].shape)}
        for i in range(L):
            d[f"present.{i}.key"] = tuple(flat[1 + 2 * i].shape)
            d[f"present.{i}.value"] = tuple(flat[2 + 2 * i].shape)
        return d

    # coherent trials: vary query len q and past len P -> empirical dynamic axes
    all_in, all_out = [], []
    trial0 = None
    for q, P in [(1, 16), (4, 8), (1, 24)]:
        emb, past = mk_inputs(q, P)
        if trial0 is None:
            trial0 = (emb, past)
        with torch.no_grad():
            flat = wrapper(emb, past)
        all_in.append(in_shapes(emb, past))
        all_out.append(out_shapes(flat))
    dyn = _compute_dynamic_axes(all_in, all_out)
    print(f"[{name}] dynamic_axes (sample): inputs_embeds={dyn.get('inputs_embeds')} "
          f"pkv0.key={dyn.get('past_key_values.0.key')} present0.key={dyn.get('present.0.key')}")

    emb0, past0 = trial0
    model_inputs = {"inputs_embeds": tuple(emb0.shape)}
    model_input_dtypes = {"inputs_embeds": emb0.dtype}
    for i, (k, v) in enumerate(past0):
        model_inputs[f"past_key_values.{i}.key"] = tuple(k.shape)
        model_inputs[f"past_key_values.{i}.value"] = tuple(v.shape)
        model_input_dtypes[f"past_key_values.{i}.key"] = k.dtype
        model_input_dtypes[f"past_key_values.{i}.value"] = v.dtype
    with torch.no_grad():
        flat0 = wrapper(emb0, past0)
    model_outputs = out_shapes(flat0)

    cfg = DummyOnnxConfig(
        config=wrapper.config, task="backbone",
        model_inputs=model_inputs, model_outputs=model_outputs,
        dynamic_axes=dyn, model_input_dtypes=model_input_dtypes, float_dtype="fp32",
    )
    # Decoder graphs leave the hidden dim symbolic via Reshape; let optimum's
    # fix_dynamic_axes run (don't disable it) so axis names resolve.
    ok = _export_and_validate(name, wrapper, cfg, output_dir, device, opset, atol,
                              disable_dynamic_axes_fix=False)
    if ok and use_fp16:
        ok = _convert_onnx_to_fp16(Path(output_dir) / name / "model.onnx")
    return ok


ALL_MODULES = list(LEAVES) + ["t3_backbone"]


def export_all(output_dir, modules=None, device="cpu", opset=17, atol=1e-3, tts=None,
               t3_dtype="fp32"):
    """Export the requested leaf modules to ``output_dir`` and return a
    ``{name: ok}`` dict. Loads ChatterboxTurboTTS if ``tts`` is not supplied.

    ``t3_dtype`` ("fp32"|"fp16") controls the t3_backbone precision (fp16 ~halves
    its size and is exported on CUDA). The other leaves are exported in fp32.

    This is the programmatic entry point used both by the CLI (``main``) and by
    ``OnTheFlyORTChatterboxPipeline.from_pretrained(export=True)``.
    """
    modules = modules or ALL_MODULES
    for name in modules:
        if name not in ALL_MODULES:
            raise ValueError(f"unknown module '{name}'. choices: {ALL_MODULES}")

    if tts is None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print(">>> loading ChatterboxTurboTTS ...", flush=True)
        tts = ChatterboxTurboTTS.from_pretrained("cuda")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in modules:
        try:
            if name == "t3_backbone":
                results[name] = export_t3_backbone(tts, output_dir, device, opset, atol,
                                                   dtype=t3_dtype)
            else:
                results[name] = export_leaf(name, tts, output_dir, device, opset, atol)
        except Exception:
            import traceback
            traceback.print_exc()
            results[name] = False
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", default=",".join(ALL_MODULES),
                    help="comma-separated leaf names: " + ",".join(ALL_MODULES))
    ap.add_argument("--output", default="/dev/shm/cbx_onnx")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--t3-dtype", default="fp32", choices=["fp32", "fp16"],
                    help="precision for t3_backbone (fp16 ~halves its size, exported on CUDA)")
    args = ap.parse_args()

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    results = export_all(args.output, modules=modules, device=args.device,
                         opset=args.opset, atol=args.atol, t3_dtype=args.t3_dtype)

    print("\n" + "=" * 70 + "\nSUMMARY")
    for k, ok in results.items():
        print(f"  {k:18s}: {'OK' if ok else 'FAILED'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
