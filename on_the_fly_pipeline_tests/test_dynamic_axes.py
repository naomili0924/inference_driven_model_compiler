"""Tests that dynamic axes are correctly inferred from multi-trial inference.

Each test:
  1. Loads a model and runs trace_model_shapes with n_trials=3.
  2. Checks that expected dynamic dims are marked dynamic.
  3. Checks that expected static dims (hidden_size, num_heads, …) are NOT dynamic.

Run with:
    PYTHONPATH=/workspace python3 \
        inference_driven_model_compiler/on_the_fly_pipeline_tests/test_dynamic_axes.py
"""
import sys
import torch
from transformers import (
    AutoTokenizer, AutoFeatureExtractor,
    BertForMaskedLM, GPT2LMHeadModel, T5Model,
)

sys.path.insert(0, "/workspace")
from inference_driven_model_compiler.optimum.exporters.onnx.utils import trace_model_shapes

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def run_suite(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    results = fn()
    ok = sum(results)
    print(f"  {ok}/{len(results)} passed")
    return ok, len(results)


# ─────────────────────────────────────────────────────────────────────────────
# BERT (encoder-only, no KV cache)
# ─────────────────────────────────────────────────────────────────────────────
def test_bert():
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()

    enc = tok("Paris is the [MASK] of France.", return_tensors="pt")
    inputs, outputs, dyn = trace_model_shapes(model, dict(enc), n_trials=3)

    results = []
    # ── inputs ────────────────────────────────────────────────────────────
    results.append(check("input_ids dim-0 (batch) is dynamic",
                          0 in dyn.get("input_ids", {})))
    results.append(check("input_ids dim-1 (seq_len) is dynamic",
                          1 in dyn.get("input_ids", {})))
    results.append(check("attention_mask dim-0 (batch) is dynamic",
                          0 in dyn.get("attention_mask", {})))
    results.append(check("attention_mask dim-1 (seq_len) is dynamic",
                          1 in dyn.get("attention_mask", {})))

    # ── outputs ───────────────────────────────────────────────────────────
    # BERT logits: (batch, seq_len, vocab_size=30522)
    logits_dyn = dyn.get("logits", {})
    results.append(check("logits dim-0 (batch) is dynamic",
                          0 in logits_dyn))
    results.append(check("logits dim-1 (seq_len) is dynamic",
                          1 in logits_dyn))
    results.append(check("logits dim-2 (vocab_size) is STATIC — not in dynamic axes",
                          2 not in logits_dyn,
                          detail=f"vocab_size=30522 should be static"))

    # hidden_size (768) should never appear as a dynamic dim
    for name, axes in dyn.items():
        for dim_idx, label in axes.items():
            shape = (inputs.get(name) or outputs.get(name))
            if shape and len(shape) > dim_idx:
                val = shape[dim_idx]
                results.append(check(
                    f"  {name}[{dim_idx}]={val} — hidden_size/vocab NOT dynamic as {label}",
                    val not in (768, 30522) or dim_idx == 0,
                ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2 (decoder-only, with KV cache)
# ─────────────────────────────────────────────────────────────────────────────
def test_gpt2():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    enc = tok("The quick brown fox", return_tensors="pt")
    inputs, outputs, dyn = trace_model_shapes(model, dict(enc), n_trials=3)

    results = []
    # decode-step inputs
    results.append(check("input_ids (decode) dim-0 (batch) is dynamic",
                          0 in dyn.get("input_ids", {})))
    results.append(check("attention_mask (decode) dim-0 (batch) is dynamic",
                          0 in dyn.get("attention_mask", {})))
    results.append(check("attention_mask (decode) dim-1 (past_seq+1) is dynamic",
                          1 in dyn.get("attention_mask", {})))

    # GPT-2 has 12 layers; check layer 0 KV
    k_name = "past_key_values.0.key"
    v_name = "past_key_values.0.value"
    results.append(check(f"{k_name} dim-0 (batch) is dynamic",
                          0 in dyn.get(k_name, {})))
    results.append(check(f"{k_name} dim-1 (num_heads=12) is STATIC",
                          1 not in dyn.get(k_name, {}),
                          detail="num_heads=12 must not be dynamic"))
    results.append(check(f"{k_name} dim-2 (past_seq) is dynamic",
                          2 in dyn.get(k_name, {})))
    results.append(check(f"{k_name} dim-3 (head_dim=64) is STATIC",
                          3 not in dyn.get(k_name, {}),
                          detail="head_dim=64 must not be dynamic"))

    # logits (decode): (batch, 1, vocab_size=50257)
    logits_dyn = dyn.get("logits", {})
    results.append(check("logits dim-0 (batch) is dynamic", 0 in logits_dyn))
    results.append(check("logits dim-2 (vocab_size=50257) is STATIC",
                          2 not in logits_dyn,
                          detail="vocab_size=50257 should be static"))

    # num_heads (12) and head_dim (64) must never appear as a dynamic dim value
    gpt2_static = {12, 64}  # num_heads, head_dim
    for name, axes in dyn.items():
        for dim_idx, label in axes.items():
            shape = inputs.get(name) or outputs.get(name)
            if shape and len(shape) > dim_idx:
                val = shape[dim_idx]
                results.append(check(
                    f"  {name}[{dim_idx}]={val} — config static not dyn",
                    val not in gpt2_static or dim_idx == 0,
                ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ViT (vision encoder, no KV cache, fixed spatial dims)
# ─────────────────────────────────────────────────────────────────────────────
def test_vit():
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    from transformers import ViTModel
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").eval()

    dummy_img = torch.randint(0, 256, (224, 224, 3)).numpy()
    enc = extractor(images=dummy_img, return_tensors="pt")
    inputs, outputs, dyn = trace_model_shapes(model, dict(enc), n_trials=3)

    results = []
    pv_dyn = dyn.get("pixel_values", {})
    results.append(check("pixel_values dim-0 (batch) is dynamic",
                          0 in pv_dyn))
    results.append(check("pixel_values dim-1 (channels=3) is STATIC",
                          1 not in pv_dyn,
                          detail="RGB channels always 3"))
    results.append(check("pixel_values dim-2 (height=224) is STATIC",
                          2 not in pv_dyn,
                          detail="fixed image height"))
    results.append(check("pixel_values dim-3 (width=224) is STATIC",
                          3 not in pv_dyn,
                          detail="fixed image width"))

    lhs_dyn = dyn.get("last_hidden_state", {})
    results.append(check("last_hidden_state dim-0 (batch) is dynamic",
                          0 in lhs_dyn))
    results.append(check("last_hidden_state dim-1 (n_patches=197) is STATIC — fixed grid",
                          1 not in lhs_dyn,
                          detail="patch count fixed at 197 for 224×224/16"))
    results.append(check("last_hidden_state dim-2 (hidden_size=768) is STATIC",
                          2 not in lhs_dyn))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# T5 encoder (encoder-decoder, only encoder exported)
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_encoder():
    tok = AutoTokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small").eval()

    enc = tok("translate English to French: Hello!", return_tensors="pt")
    inputs, outputs, dyn = trace_model_shapes(model, dict(enc), n_trials=3)

    results = []
    results.append(check("input_ids dim-0 (batch) is dynamic",
                          0 in dyn.get("input_ids", {})))
    results.append(check("input_ids dim-1 (seq_len) is dynamic",
                          1 in dyn.get("input_ids", {})))

    lhs_dyn = dyn.get("last_hidden_state", {})
    results.append(check("last_hidden_state dim-0 (batch) is dynamic",
                          0 in lhs_dyn))
    results.append(check("last_hidden_state dim-1 (seq_len) is dynamic",
                          1 in lhs_dyn))
    results.append(check("last_hidden_state dim-2 (d_model=512) is STATIC",
                          2 not in lhs_dyn,
                          detail="T5-small d_model=512 is a config constant"))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_ok, total_n = 0, 0
    for name, fn in [
        ("BERT (encoder-only, no KV cache)", test_bert),
        ("GPT-2 (decoder-only, KV cache)", test_gpt2),
        ("ViT (vision encoder, fixed spatial dims)", test_vit),
        ("T5 encoder (encoder-decoder)", test_t5_encoder),
    ]:
        ok, n = run_suite(name, fn)
        total_ok += ok
        total_n  += n

    print(f"\n{'='*60}")
    print(f"  Total: {total_ok}/{total_n} passed")
    print(f"{'='*60}")
    sys.exit(0 if total_ok == total_n else 1)
