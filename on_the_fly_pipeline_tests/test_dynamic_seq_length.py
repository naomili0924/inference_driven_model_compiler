"""End-to-end test: the exported ONNX model runs at sequence lengths and
batch sizes different from the one it was traced with.

This proves the empirically-inferred dynamic axes actually work at runtime
(not merely that they are labelled dynamic in the ONNX config).

Run with:
    PYTHONPATH=/workspace python3 \
        inference_driven_model_compiler/on_the_fly_pipeline_tests/test_dynamic_seq_length.py
"""
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace")
from inference_driven_model_compiler.optimum.onnxruntime import (
    OnTheFlyORTModelForFeatureExtraction,
    OnTheFlyORTModelForMaskedLM,
    OnTheFlyORTModelForCausalLM,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def run_suite(name, fn):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")
    results = fn()
    print(f"  {sum(results)}/{len(results)} passed")
    return sum(results), len(results)


# ─────────────────────────────────────────────────────────────────────────────
# BERT feature-extraction: vary sequence length AND batch size
# ─────────────────────────────────────────────────────────────────────────────
def test_bert_variable_shapes():
    ckpt = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)

    # Trace/export with ONE short prompt (seq ~6)
    trace_inp = tok("The cat sat.", return_tensors="pt")
    model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
        ckpt,
        inference_kwargs=dict(trace_inp),
        export_by_inference=True,
        export=True,
        module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    )
    hidden = model.config.hidden_size

    results = []
    # Run inference at a variety of (batch, seq) shapes — all different from trace
    test_prompts = [
        ["A much longer sentence than the one used for tracing, with many tokens."],
        ["short one", "another short one", "a third sentence here"],          # batch 3
        ["tiny"],                                                              # very short
        ["padded batch sentence one that is fairly long indeed",
         "second sentence"],                                                  # batch 2, padded
    ]
    for prompt in test_prompts:
        enc = tok(prompt, return_tensors="pt", padding=True)
        out = model(**enc)
        exp_batch, exp_seq = enc["input_ids"].shape
        got = tuple(out.last_hidden_state.shape)
        ok = got == (exp_batch, exp_seq, hidden)
        results.append(check(
            f"batch={exp_batch}, seq={exp_seq} → last_hidden_state {got}",
            ok,
            detail="" if ok else f"expected ({exp_batch}, {exp_seq}, {hidden})",
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# BERT masked-LM: logits scale with sequence length, vocab stays fixed
# ─────────────────────────────────────────────────────────────────────────────
def test_bert_mlm_variable_seq():
    ckpt = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(ckpt)

    trace_inp = tok("Paris is the [MASK] of France.", return_tensors="pt")
    model = OnTheFlyORTModelForMaskedLM.from_pretrained(
        ckpt,
        inference_kwargs=dict(trace_inp),
        export_by_inference=True,
        export=True,
        module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
    )
    vocab = model.config.vocab_size

    results = []
    for text in [
        "The [MASK] is bright today.",
        "I went to the [MASK] yesterday to buy some groceries and then came home.",
        "[MASK].",
    ]:
        enc = tok(text, return_tensors="pt")
        out = model(**enc)
        b, s = enc["input_ids"].shape
        got = tuple(out.logits.shape)
        ok = got == (b, s, vocab)
        results.append(check(
            f"seq={s} → logits {got}", ok,
            detail="" if ok else f"expected ({b}, {s}, {vocab})",
        ))

    # mask prediction should still be sensible after re-shaping
    enc = tok("Paris is the [MASK] of France.", return_tensors="pt")
    out = model(**enc)
    mask_pos = (enc["input_ids"] == tok.mask_token_id)[0].nonzero(as_tuple=True)[0]
    pred = tok.decode(out.logits[0, mask_pos].argmax(-1)).strip()
    results.append(check(f"mask prediction = '{pred}'", pred == "capital",
                         detail="expected 'capital'"))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GPT-2: generation works at prompt lengths different from the trace
# (exercises both dynamic prefill seq and dynamic KV past-length)
# ─────────────────────────────────────────────────────────────────────────────
def test_gpt2_variable_generation():
    ckpt = "gpt2"
    tok = AutoTokenizer.from_pretrained(ckpt)

    trace_inp = tok("Hello world", return_tensors="pt")     # 2-token trace
    model = OnTheFlyORTModelForCausalLM.from_pretrained(
        ckpt,
        inference_kwargs=dict(trace_inp),
        export_by_inference=True,
        export=True,
        module_fixed_axis_fields={"transformer": ["n_ctx", "n_embd"]},
    )

    results = []
    cases = [
        ("Short prompt", 8),
        ("A considerably longer prompt with several more words than the trace input", 16),
        ("Mid", 24),
    ]
    for prompt, max_new in cases:
        enc = tok(prompt, return_tensors="pt")
        in_len = enc["input_ids"].shape[1]
        out_ids = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        out_len = out_ids.shape[1]
        ok = out_len > in_len                       # produced new tokens
        results.append(check(
            f"prompt_len={in_len}, max_new={max_new} → generated {out_len} tokens",
            ok,
            detail="" if ok else "no new tokens generated",
        ))

    # different prompt lengths must give different prefill seq → still valid
    a = model.generate(**tok("One", return_tensors="pt"), max_new_tokens=5, do_sample=False)
    b = model.generate(**tok("One two three four five", return_tensors="pt"),
                       max_new_tokens=5, do_sample=False)
    results.append(check("two different prompt lengths both generate",
                         a.shape[1] > 1 and b.shape[1] > 5))
    return results


if __name__ == "__main__":
    total_ok, total_n = 0, 0
    for name, fn in [
        ("BERT feature-extraction — variable batch & seq", test_bert_variable_shapes),
        ("BERT masked-LM — variable seq, fixed vocab", test_bert_mlm_variable_seq),
        ("GPT-2 — variable prompt length generation", test_gpt2_variable_generation),
    ]:
        ok, n = run_suite(name, fn)
        total_ok += ok
        total_n  += n

    print(f"\n{'='*64}\n  Total: {total_ok}/{total_n} passed\n{'='*64}")
    sys.exit(0 if total_ok == total_n else 1)
