# Inference-Driven Model Compiler

Export 🤗 Transformers models to ONNX by **observing a real inference pass**
instead of relying on hand-written, per-architecture ONNX configurations.

Standard [Optimum](https://github.com/huggingface/optimum) ONNX export requires
a model-specific `OnnxConfig` that declares every input/output and which tensor
dimensions are dynamic. This project removes that requirement: it runs the model
on your actual inputs, traces the tensor shapes that flow through it, determines
which dimensions are dynamic **empirically**, and exports the result — all behind
the familiar `from_pretrained(...)` interface.

It is built entirely on top of an **unmodified** `optimum` / `optimum-onnx`
installation.

---

## Why

| Standard Optimum export | Inference-driven export |
|---|---|
| Needs a hand-written `OnnxConfig` per architecture | Works with any model that runs a forward pass |
| Dynamic axes declared manually | Dynamic axes inferred from multiple varied runs |
| New architectures require code changes upstream | New architectures work out of the box |

---

## How it works

```
from_pretrained(export_by_inference=True)
        │
        ▼
1. Load the PyTorch model (TasksManager)
        │
        ▼
2. trace_model_shapes()  ── run N inference passes with varied input shapes
        │                    • encoder-only      → single forward
        │                    • encoder-decoder   → encoder submodule only
        │                    • decoder-only      → prefill + decode (KV cache)
        ▼
3. _compute_dynamic_axes()  ── a dim is "dynamic" iff its size changed across runs
        │                       (dim-0/batch always dynamic; hidden_size, vocab,
        ▼                        num_heads, head_dim, image H/W … stay static)
4. DummyOnnxConfig  ── a generic OnnxConfig built from the traced shapes + axes
        │
        ▼
5. export_models()  ── standard Optimum ONNX export (disable_dynamic_axes_fix=True)
        │
        ▼
6. ORTModel._from_pretrained()  ── load the ONNX model into an ORT session
```

### Dynamic-axis inference

Rather than guessing from config fields, the compiler runs the model several
times (default `n_trials=3`) with randomly varied **batch sizes** and **sequence
lengths** (and, for decoder models, a varied **decode query length**). A
dimension is marked dynamic only if its value actually changes between runs;
everything else — `hidden_size`, `num_attention_heads`, `head_dim`,
`vocab_size`, image height/width, ViT patch count, etc. — is correctly kept
static.

---

## Installation

```bash
pip install torch transformers onnx onnxruntime
pip install "optimum @ git+https://github.com/huggingface/optimum"

# clone this repo so that `inference_driven_model_compiler` is importable
git clone https://github.com/naomili0924/inference_driven_model_compiler.git
export PYTHONPATH=/path/to/parent_of_repo:$PYTHONPATH
```

---

## Quick start

### Encoder model (BERT feature extraction)

```python
from transformers import AutoTokenizer
from inference_driven_model_compiler.optimum.onnxruntime import (
    OnTheFlyORTModelForFeatureExtraction,
)

ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
encoded = tokenizer("ONNX Runtime accelerates inference.", return_tensors="pt")

model = OnTheFlyORTModelForFeatureExtraction.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded),     # the inputs to trace
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["hidden_size", "num_attention_heads"]},
)

out = model(**encoded)
print(out.last_hidden_state.shape)      # (1, seq_len, 768)
```

### Decoder model (GPT-2 text generation, with KV cache)

```python
from transformers import GPT2Tokenizer
from inference_driven_model_compiler.optimum.onnxruntime import OnTheFlyORTModelForCausalLM

ckpt = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(ckpt)
encoded = tokenizer("Replace me by any text you'd like.", return_tensors="pt")

model = OnTheFlyORTModelForCausalLM.from_pretrained(
    ckpt,
    inference_kwargs=dict(encoded),
    export_by_inference=True,
    export=True,
    module_fixed_axis_fields={"transformer": ["n_ctx", "n_embd"]},
)

output_ids = model.generate(**encoded)
print(tokenizer.decode(output_ids[0]))
```

### Common arguments

| Argument | Meaning |
|---|---|
| `inference_kwargs` | The inputs used to trace the model (e.g. a tokenized prompt). |
| `export_by_inference=True` | Enable the inference-driven export path. |
| `export=True` | Force an ONNX export (vs. loading an existing one). |
| `module_fixed_axis_fields` | Config fields whose values should be treated as fixed dims (a hint for the static/dynamic heuristic fallback). |
| `skip_random_generation` | Keep the actual traced tensors as fixed dummy inputs instead of regenerating them. |
| `n_trials` | Number of inference passes for dynamic-axis detection (default `3`). |

---

## Available classes

All live in `inference_driven_model_compiler.optimum.onnxruntime` and share the
same `from_pretrained(...)` interface:

| Class | Task |
|---|---|
| `OnTheFlyORTModelForCausalLM` | Text generation (decoder-only, KV cache) |
| `OnTheFlyORTModelForFeatureExtraction` | Embeddings / hidden states |
| `OnTheFlyORTModelForMaskedLM` | Masked language modeling |
| `OnTheFlyORTModelForSequenceClassification` | Sequence classification |
| `OnTheFlyORTModelForTokenClassification` | Token classification / NER |
| `OnTheFlyORTModelForQuestionAnswering` | Extractive QA |

---

## Verified models

| Model | Type | Task tested |
|---|---|---|
| GPT-2 | decoder-only | text generation (KV cache) |
| BERT-base | encoder | masked-LM, seq-cls, token-cls, QA, feature extraction |
| T5-small | encoder-decoder | feature extraction (encoder) |
| BART-base | encoder-decoder | feature extraction (encoder) |
| ViT-base | vision encoder | feature extraction |
| CLIP-ViT-base | vision encoder | feature extraction |
| Whisper-tiny | audio encoder | feature extraction |

---

## Tests

```bash
cd /path/to/parent_of_repo

# Per-model end-to-end export + inference
PYTHONPATH=. python3 inference_driven_model_compiler/on_the_fly_pipeline_tests/gpt2_text_generation.py
PYTHONPATH=. python3 inference_driven_model_compiler/on_the_fly_pipeline_tests/bert_masked_lm.py
# … etc.

# Dynamic-axis inference correctness (structural — 140 checks)
PYTHONPATH=. python3 inference_driven_model_compiler/on_the_fly_pipeline_tests/test_dynamic_axes.py

# Dynamic sequence-length inference (runtime — 12 checks)
PYTHONPATH=. python3 inference_driven_model_compiler/on_the_fly_pipeline_tests/test_dynamic_seq_length.py
```

- **`test_dynamic_axes.py`** — asserts each dimension is labelled dynamic/static
  correctly for BERT, GPT-2, ViT and T5 (batch & seq dynamic; hidden_size,
  num_heads, head_dim, vocab, patch-count static).
- **`test_dynamic_seq_length.py`** — exports each model once, then runs the
  ONNX model at sequence lengths and batch sizes **different from the trace**,
  proving the dynamic axes work at runtime.

---

## Project layout

```
inference_driven_model_compiler/
├── optimum/
│   ├── exporters/onnx/
│   │   ├── utils.py            # trace_model_shapes(), dynamic-axis inference
│   │   ├── model_configs.py    # DummyOnnxConfig — generic shape-driven OnnxConfig
│   │   ├── input_generators.py # DummyTupleInputGenerator — dtype-aware dummies
│   │   └── __init__.py          # main_export wrapper (renamed params)
│   └── onnxruntime/
│       ├── modeling.py          # _OnTheFlyORTMixin + 5 encoder model classes
│       └── modeling_decoder.py  # OnTheFlyORTModelForCausalLM
├── on_the_fly_pipeline_tests/   # per-model tests + dynamic-axis test suites
└── baseline_pipeline_tests/     # plain Optimum baselines for comparison
```

---

## How dynamic vs. static axes are decided

For each tensor seen across the trial runs:

- **Dimension 0** → always dynamic (`batch`).
- **Any other dimension** → dynamic **iff** its size differed between at least
  two trials; otherwise static.
- For decoder KV-cache tensors `past_key_values.{i}.key/value`, the
  past-sequence dimension (axis 2) is dynamic while `num_heads` (axis 1) and
  `head_dim` (axis 3) stay static.

This means architecture constants pulled from `model.config`
(`hidden_size`, `vocab_size`, `num_attention_heads`, image resolution, …) are
never accidentally exported as dynamic.

---

## Limitations

- Encoder-decoder models (T5, BART, Whisper) are exported **encoder-only** for
  feature-extraction; full encoder-decoder generation is not yet wired up.
- CLIP exports the **vision encoder** (the full CLIP forward needs both text and
  image inputs and returns embeddings rather than `last_hidden_state`).
- The exported ONNX is written to a temporary directory; call
  `model.save_pretrained(...)` to persist it.
