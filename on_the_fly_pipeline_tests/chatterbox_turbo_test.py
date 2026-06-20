"""OnTheFlyORTChatterboxPipeline.from_pretrained smoke + parity test (chatterbox-turbo TTS).

Loads a pre-exported chatterbox-turbo ONNX bundle (the 4 leaf graphs produced by
``chatterbox_export.py``) through ``OnTheFlyORTChatterboxPipeline.from_pretrained`` and:

  1. SMOKE   — runs generate() and checks the audio is finite and non-empty.
  2. PARITY  — greedy-decodes the T3 backbone in PyTorch vs ONNX with identical
               conditioning/seed and asserts the speech-token sequences match
               (the ONNX GPT2-with-past backbone must be faithful in the real
               autoregressive loop), and that the S3Gen (estimator + HiFiGAN)
               path matches PyTorch on shared tokens within a small tolerance.

``--source`` accepts a local export dir (default ``/dev/shm/cbx_onnx``) or a Hub
repo id (e.g. ``Jinyan0924/chatterbox-turbo-onnx``; pass ``--token`` if private).

Run (needs the chatterbox venv + a GPU):

    PYTHONPATH=/workspace HF_HOME=/dev/shm/hf \
        python inference_driven_model_compiler/on_the_fly_pipeline_tests/chatterbox_turbo_test.py \
        --source /dev/shm/cbx_onnx
"""
import argparse
import sys

import numpy as np
import torch

from inference_driven_model_compiler.ort_chatterbox import OnTheFlyORTChatterboxPipeline


def _greedy_tokens(tts, text, seed=0, max_gen_len=150):
    from chatterbox.tts_turbo import punc_norm
    torch.manual_seed(seed)
    ids = tts.tokenizer(punc_norm(text), return_tensors="pt", padding=True,
                        truncation=True).input_ids.to(tts.device)
    # top_k=1 -> deterministic argmax decode (no temperature/top_p/rep-penalty effect)
    return tts.t3.inference_turbo(t3_cond=tts.conds.t3, text_tokens=ids, temperature=1.0,
                                  top_k=1, top_p=1.0, repetition_penalty=1.0,
                                  max_gen_len=max_gen_len)


def _s3gen_wav(tts, tokens, seed=42):
    from chatterbox.tts_turbo import S3GEN_SIL
    torch.manual_seed(seed)
    st = tokens[tokens < 6561]
    st = torch.cat([st, torch.tensor([S3GEN_SIL] * 3, device=tts.device).long()])
    wav, _ = tts.s3gen.inference(speech_tokens=st, ref_dict=tts.conds.gen, n_cfm_timesteps=2)
    return wav.squeeze(0).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/dev/shm/cbx_onnx",
                    help="local export dir or HF repo id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token", default=None, help="HF token (private repos)")
    ap.add_argument("--text", default="Hello from ONNX Runtime, this is a parity test.")
    ap.add_argument("--no-parity", action="store_true", help="skip the torch-vs-ONNX parity check")
    ap.add_argument("--s3gen-atol", type=float, default=5e-2)
    args = ap.parse_args()

    failures = []

    # ---- load via from_pretrained ----
    print(f">>> OnTheFlyORTChatterboxPipeline.from_pretrained({args.source!r}, device={args.device!r})")
    pipe = OnTheFlyORTChatterboxPipeline.from_pretrained(args.source, device=args.device, token=args.token)

    # ---- 1. SMOKE ----
    wav = pipe.generate(args.text)
    w = wav.detach().cpu().numpy()
    dur = w.shape[-1] / pipe.sr
    ok = bool(np.isfinite(w).all()) and w.shape[-1] > 0 and wav.dim() == 2
    print(f"[smoke] wav={tuple(wav.shape)} sr={pipe.sr} dur={dur:.2f}s "
          f"finite={np.isfinite(w).all()} range=[{w.min():.3f},{w.max():.3f}] -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("smoke")

    # ---- 2. PARITY vs PyTorch ----
    if not args.no_parity:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        ref = ChatterboxTurboTTS.from_pretrained(args.device)  # pure-torch reference

        toks_ref = _greedy_tokens(ref, args.text)
        wav_ref = _s3gen_wav(ref, toks_ref.clone())
        del ref
        if args.device == "cuda":
            torch.cuda.empty_cache()

        toks_ort = _greedy_tokens(pipe.tts, args.text)
        wav_ort = _s3gen_wav(pipe.tts, toks_ref.clone())  # same tokens as ref

        a, b = toks_ref[0].tolist(), toks_ort[0].tolist()
        t3_ok = (a == b)
        lcp = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"[parity t3] torch {len(a)} toks vs onnx {len(b)} toks | identical={t3_ok} "
              f"| common_prefix={lcp}/{min(len(a), len(b))} -> {'PASS' if t3_ok else 'FAIL'}")
        if not t3_ok:
            failures.append("parity-t3")

        m = min(len(wav_ref), len(wav_ort))
        mad = float(np.abs(wav_ref[:m] - wav_ort[:m]).max())
        s3_ok = (len(wav_ref) == len(wav_ort)) and mad < args.s3gen_atol
        print(f"[parity s3gen] len torch {len(wav_ref)} onnx {len(wav_ort)} | "
              f"max|diff|={mad:.2e} (atol {args.s3gen_atol}) -> {'PASS' if s3_ok else 'FAIL'}")
        if not s3_ok:
            failures.append("parity-s3gen")

    print("\nRESULT:", "ALL PASS" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
