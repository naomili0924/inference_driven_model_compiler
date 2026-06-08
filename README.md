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

## Installation

```bash
pip install torch transformers onnx onnxruntime
pip install "optimum @ git+https://github.com/huggingface/optimum"
# the ONNX exporter/runtime now lives in the separate optimum-onnx package
pip install "optimum-onnx[onnxruntime] @ git+https://github.com/huggingface/optimum-onnx"

git clone https://github.com/naomili0924/inference_driven_model_compiler.git
# Put the *repo directory itself* on PYTHONPATH. This activates the shadow
# `optimum` package, which (a) makes `optimum-cli` use the inference-driven
# exporter and (b) auto-registers the new export flags onto
# `optimum-cli export onnx` (see optimum/commands/register/register_idmc.py).
# Add the parent dir too if you also want `import inference_driven_model_compiler`
# or the `idmc` CLI.
export PYTHONPATH=/path/to/inference_driven_model_compiler:$PYTHONPATH
```

With this repo *off* PYTHONPATH, `optimum-cli` behaves exactly as stock — the
integration is inert unless the shadow `optimum` is active.

---

## CLI export

When this repo is on `PYTHONPATH` (see Installation), the standard
`optimum-cli export onnx` command gains three extra flags — no separate tool or
launcher needed. They are registered automatically via
`optimum/commands/register/register_idmc.py`, which optimum-cli auto-discovers:

| Flag | Description |
|---|---|
| `--export_by_inference` | Enable inference-driven export (traces the model instead of using a hand-written `OnnxConfig`). |
| `--module_fixed_axis_fields` | JSON dict mapping submodule names to config field names whose values should be treated as **static** tensor dimensions. |
| `--inference_kwargs` | JSON dict of inputs used to trace the model (overrides the auto-generated dummy inputs). |

### Encoder model

```bash
optimum-cli export onnx \
    --model sentence-transformers/paraphrase-MiniLM-L12-v2 \
    /dev/shm/paraphrase-MiniLM \
    --export_by_inference=true \
    --module_fixed_axis_fields='{"transformer": ["hidden_size","intermediate_size","type_vocab_size","vocab_size"]}'
```

### Decoder model (with KV cache)

```bash
optimum-cli export onnx \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    /dev/shm/qwen3-4b-thinking-onnx \
    --task text-generation-with-past \
    --export_by_inference=true \
    --dtype fp16 --device cpu
```

> The inference-driven tracer builds its dummy inputs on CPU, so export decoder
> models with `--device cpu` (a `--device cuda` model would mismatch the
> CPU-resident traced inputs).

`--module_fixed_axis_fields` is optional for decoder models — the dynamic-axis
inference step figures out `num_heads`, `head_dim`, etc. automatically.

> **Tip for large models:** if your disk is limited, export to `/dev/shm` (a
> RAM-backed tmpfs typically >= 80 GB on GPU instances) and copy the result
> elsewhere afterwards.

---

## Python API

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
    inference_kwargs=dict(encoded),
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

### Diffusion pipeline (text-to-video)

Pass `export_by_inference=True` together with `inference_kwargs` (the same kwargs
you would pass to the pipeline `__call__`). The pipeline runs once in PyTorch to
capture real tensor shapes for every submodule, then exports each one to ONNX
automatically — no hand-written `OnnxConfig` required.

```python
import torch
from inference_driven_model_compiler.optimum.onnxruntime import ORTDiffusionPipeline

inf_kwargs = {
    "prompt": "A cat walks on the grass, realistic",
    "negative_prompt": "low quality, blurred",
    "height": 240,
    "width": 416,
    "num_frames": 21,
    "guidance_scale": 5.0,
}

pipe = ORTDiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    provider="CUDAExecutionProvider",
    torch_dtype=torch.float16,
    export_by_inference=True,
    inference_kwargs=inf_kwargs,
    module_fixed_axis_fields={
        "text_encoder": ["d_model", "vocab_size"],
        "transformer":  ["in_channels", "text_dim"],
        "vae_decoder":  ["base_dim", "z_dim"],
    },
)

output = pipe(**inf_kwargs).frames[0]
```

This exports three ONNX files to a temporary directory and immediately loads them
into ORT sessions — all in one `from_pretrained` call:

| Submodule | ONNX file | Typical size |
|---|---|---|
| `text_encoder` | `text_encoder/model.onnx` | ~13 GB (fp16) |
| `transformer` | `transformer/model.onnx` | ~3 GB (fp16) |
| `vae_decoder` | `vae_decoder/model.onnx` | ~137 MB (fp16) |

### Loading pre-exported ONNX weights

```python
pipe = ORTDiffusionPipeline.from_pretrained(
    "optimum/stable-diffusion-v1-5",   # Hub repo with pre-exported ONNX weights
    export=False,
)
```

### Common arguments

| Argument | Meaning |
|---|---|
| `inference_kwargs` | Inputs used to trace the model (e.g. a tokenized prompt, or full pipeline kwargs for diffusion). |
| `export_by_inference=True` | Enable the inference-driven export path. |
| `export=True` | Force a fresh ONNX export (transformer models). |
| `module_fixed_axis_fields` | Per-submodule config field names whose values should be treated as fixed (static) tensor dims. |
| `skip_random_generation` | Keep the actual traced tensors as fixed dummy inputs instead of regenerating them. |
| `n_trials` | Number of inference passes for dynamic-axis detection, transformer models only (default `3`). |

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

Supported text-to-video pipeline names (as of diffusers 0.38):
`AnimateDiffPipeline`, `AnimateDiffSDXLPipeline`, `CogVideoXPipeline`,
`HunyuanVideo15Pipeline`, `HunyuanVideoPipeline`, `LTXPipeline`, `LTX2Pipeline`,
`LattePipeline`, `MochiPipeline`, `SanaVideoPipeline`, `TextToVideoSDPipeline`,
`WanPipeline`, `WanAnimatePipeline`.

---

## Verified models

### Transformer models

| Model | Type | Task tested |
|---|---|---|
| GPT-2 | decoder-only | text generation (KV cache) |
| Gemma 4 (2B) | decoder-only | text generation (KV cache, fp16) |
| BERT-base | encoder | masked-LM, seq-cls, token-cls, QA, feature extraction |
| Sentence-Transformers / paraphrase-MiniLM-L12-v2 | encoder | feature extraction |
| T5-small | encoder-decoder | feature extraction (encoder) |
| BART-base | encoder-decoder | feature extraction (encoder) |
| ViT-base | vision encoder | feature extraction |
| CLIP-ViT-base | vision encoder | feature extraction |
| Whisper-tiny | audio encoder | feature extraction |

### Diffusion pipelines

| Model | Pipeline | Submodules exported | Notes |
|---|---|---|---|
| Wan2.1-T2V-1.3B | `WanPipeline` | text_encoder, transformer, vae_decoder | Verified end-to-end on CUDA; 50-step inference at ~7.4 it/s |

---

## Limitations

- Encoder-decoder models (T5, BART, Whisper) are exported **encoder-only** for
  feature-extraction; full encoder-decoder generation is not yet wired up.
- CLIP exports the **vision encoder** (the full CLIP forward needs both text and
  image inputs and returns embeddings rather than `last_hidden_state`).
- The exported ONNX is written to a temporary directory; call
  `model.save_pretrained(...)` to persist it.
- Diffusion pipeline export runs one full inference pass before exporting, which
  requires enough GPU/CPU memory to hold the full PyTorch pipeline during tracing.
- VAE encoder export is included in the export spec but the WAN pipeline does not
  use it during text-to-video inference; it is exported as a no-op placeholder
  when the submodule exists on the VAE.

---

## How it works

### Transformer models

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

### Diffusion pipelines

```
ORTDiffusionPipeline.from_pretrained(export_by_inference=True, inference_kwargs={...})
        │
        ▼
1. Load the PyTorch diffusion pipeline (diffusers)
        │
        ▼
2. Register forward pre-hooks on each submodule
   (text_encoder, transformer/unet, vae.post_quant_conv)
        │
        ▼
3. Run ONE full pipeline inference pass with the provided inference_kwargs
   — all submodule inputs are captured live as real tensors
        │
        ▼
4. For each submodule:
   • Move captured tensors to CPU
   • Build DummyOnnxConfig from the observed shapes + dynamic axes
   • Export to ONNX (constant folding disabled for text encoders to
     avoid 89 GB inflation from precomputed attention bias)
        │
        ▼
5. VAE decoder special case: export post_quant_conv + decoder as a single
   _VaeFullDecodeWrapper (WAN VAE decodes per-frame with caching in PyTorch;
   ONNX needs one call over the full latent video)
        │
        ▼
6. ORTDiffusionPipeline loaded with each ONNX submodule in an ORT session
```

### Dynamic-axis inference

Rather than guessing from config fields, the compiler runs the model several
times (default `n_trials=3`) with randomly varied **batch sizes** and **sequence
lengths** (and, for decoder models, a varied **decode query length**). A
dimension is marked dynamic only if its value actually changes between runs;
everything else — `hidden_size`, `num_attention_heads`, `head_dim`,
`vocab_size`, image height/width, ViT patch count, etc. — is correctly kept
static.

For each tensor seen across the trial runs:

- **Dimension 0** → always dynamic (`batch`).
- **Any other dimension** → dynamic **iff** its size differed between at least
  two trials; otherwise static.
- For decoder KV-cache tensors `past_key_values.{i}.key/value`, the
  past-sequence dimension (axis 2) is dynamic while `num_heads` (axis 1) and
  `head_dim` (axis 3) stay static.

---

## Why

| Standard Optimum export | Inference-driven export |
|---|---|
| Needs a hand-written `OnnxConfig` per architecture | Works with any model that runs a forward pass |
| Dynamic axes declared manually | Dynamic axes inferred from multiple varied runs |
| New architectures require code changes upstream | New architectures work out of the box |

---

## Project layout

```
inference_driven_model_compiler/
├── cli.py                        # idmc CLI — wraps optimum-cli with extra flags
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
└── on_the_fly_pipeline_tests/    # per-model tests + dynamic-axis + diffusion suites
```
