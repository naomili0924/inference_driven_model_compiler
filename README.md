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

### Transformer models

| Class | Task |
|---|---|
| `OnTheFlyORTModelForCausalLM` | Text generation (decoder-only, KV cache) |
| `OnTheFlyORTModelForFeatureExtraction` | Embeddings / hidden states |
| `OnTheFlyORTModelForMaskedLM` | Masked language modeling |
| `OnTheFlyORTModelForSequenceClassification` | Sequence classification |
| `OnTheFlyORTModelForTokenClassification` | Token classification / NER |
| `OnTheFlyORTModelForQuestionAnswering` | Extractive QA |

### Diffusion pipelines

| Class | Purpose |
|---|---|
| `ORTDiffusionPipeline` | Generic base — wraps **any** `diffusers.DiffusionPipeline` |
| `ORTUnet` | ORT session wrapper for a UNet2D/3D denoiser |
| `ORTTransformer` | ORT session wrapper for a DiT/transformer denoiser |
| `ORTTextEncoder` | ORT session wrapper for a text encoder |
| `ORTVaeEncoder` | ORT session wrapper for a VAE encoder |
| `ORTVaeDecoder` | ORT session wrapper for a VAE decoder |
| `ORTVae` | Combines `ORTVaeEncoder` + `ORTVaeDecoder` behind the standard `vae` API |

`ORTDiffusionPipeline` requires no model-specific subclass. When called as the
base class it reads `_class_name` from the model's `model_index.json` and creates
an `ORT<ClassName>` wrapper on the fly via `_make_ort_pipeline_class`. Every
diffusers pipeline — including ones not yet written — is handled automatically.

```python
from inference_driven_model_compiler.optimum.onnxruntime import ORTDiffusionPipeline

# Export a Stable Diffusion pipeline to ONNX and load it in one call
pipe = ORTDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    export=True,            # export the PyTorch model to ONNX first
    provider="CUDAExecutionProvider",
)
image = pipe("a photo of an astronaut riding a horse").images[0]
```

For a pipeline that is already exported (or downloaded from the Hub with ONNX
weights):

```python
pipe = ORTDiffusionPipeline.from_pretrained(
    "optimum/stable-diffusion-v1-5",   # Hub repo with pre-exported ONNX weights
    export=False,
)
```

#### Text-to-video pipelines

Every text-to-video pipeline in `diffusers` works without any new code:

```python
# Wan — no ORTWanPipeline class needed
pipe = ORTDiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", export=True)

# CogVideoX
pipe = ORTDiffusionPipeline.from_pretrained("THUDM/CogVideoX-2b", export=True)

# HunyuanVideo
pipe = ORTDiffusionPipeline.from_pretrained("tencent/HunyuanVideo", export=True)
```

The complete list of verified text-to-video pipeline names (as of diffusers 0.38):
`AnimateDiffPipeline`, `AnimateDiffSDXLPipeline`, `CogVideoXPipeline`,
`HunyuanVideo15Pipeline`, `HunyuanVideoPipeline`, `LTXPipeline`, `LTX2Pipeline`,
`LattePipeline`, `MochiPipeline`, `SanaVideoPipeline`, `TextToVideoSDPipeline`,
`WanPipeline`, `WanAnimatePipeline`.

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
cd /workspace   # must be the parent of the repo — avoids optimum/ shadowing

# Run all tests (14 transformer tests + 1 diffusion test suite)
bash inference_driven_model_compiler/run_tests.sh

# Run a single file
TESTS="bert_feature_extraction.py" bash inference_driven_model_compiler/run_tests.sh

# Set timeout per test (default 600 s)
TIMEOUT=120 bash inference_driven_model_compiler/run_tests.sh
```

### Transformer model tests (`on_the_fly_pipeline_tests/`)

| File | What it tests |
|---|---|
| `bert_feature_extraction.py` | BERT encoder → feature extraction |
| `bert_masked_lm.py` | BERT masked-LM |
| `bert_sequence_classification.py` | BERT sequence classification |
| `bert_token_classification.py` | BERT NER |
| `bert_qa.py` | BERT extractive QA |
| `gpt2_text_generation.py` | GPT-2 text generation with KV cache |
| `t5_feature_extraction.py` | T5 encoder feature extraction |
| `bart_feature_extraction.py` | BART encoder feature extraction |
| `vit_feature_extraction.py` | ViT vision encoder |
| `clip_feature_extraction.py` | CLIP vision encoder |
| `whisper_feature_extraction.py` | Whisper audio encoder |
| `sentence_transformer_feature_extraction.py` | Sentence-Transformers BERT |
| `test_dynamic_axes.py` | 140 structural checks: each dim is labelled dynamic/static correctly for BERT, GPT-2, ViT, T5 |
| `test_dynamic_seq_length.py` | 12 runtime checks: exported ONNX runs at shapes different from the trace |

### Diffusion pipeline tests (`on_the_fly_pipeline_tests/test_diffusion_pipeline.py`)

43 unit tests — no GPU or model download required (all sessions are mocked):

- `TestMakeORTPipelineClass` — `_make_ort_pipeline_class` for each of the 13 text-to-video pipelines
- `TestDynamicClassCoverage` — future pipelines, MRO order, no missing pipeline names
- `TestTextToVideoPipelineClasses` — explicit per-pipeline regression guard
- `TestFromPretrainedMocked` — `from_pretrained` class-selection, subclass identity, unknown-name error
- `TestSubmoduleForward` — `ORTTransformer`, `ORTTextEncoder`, `ORTVaeEncoder`, `ORTVaeDecoder`, `ORTUnet` forward with mocked sessions
- `TestIOBindingWiring` — `set_io_binding_file`, `load_shapes_as_torch_size`
- `TestORTVae` — encode/decode, encoder-only, decoder-only

---

## Project layout

```
inference_driven_model_compiler/
├── optimum/
│   ├── exporters/onnx/
│   │   ├── utils.py              # trace_model_shapes(), dynamic-axis inference
│   │   ├── model_configs.py      # DummyOnnxConfig — generic shape-driven OnnxConfig
│   │   ├── input_generators.py   # DummyTupleInputGenerator — dtype-aware dummies
│   │   └── __init__.py           # main_export wrapper (renamed params)
│   └── onnxruntime/
│       ├── modeling.py           # _OnTheFlyORTMixin + 5 encoder model classes
│       ├── modeling_decoder.py   # OnTheFlyORTModelForCausalLM
│       ├── modeling_diffusion.py # ORTDiffusionPipeline + submodule wrappers
│       └── utils.py              # load_shapes_as_torch_size and helpers
├── on_the_fly_pipeline_tests/    # per-model tests + dynamic-axis + diffusion suites
├── baseline_pipeline_tests/      # plain Optimum baselines for comparison
└── run_tests.sh                  # test runner (run from /workspace)
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
