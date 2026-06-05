"""
CLI for inference-driven ONNX export of diffusion and transformer pipelines.

Usage
-----
After `pip install -e .`:

    idmc export --model Wan-AI/Wan2.1-T2V-1.3B --output ./wan_onnx \
                --prompt "a cat walking in the rain" --height 480 --width 832 \
                --num-frames 49 --num-inference-steps 1

    idmc export --model bert-base-uncased --task text-classification \
                --output ./bert_onnx --seq-length 128

Pass any additional pipeline kwargs as JSON with --extra-kwargs:

    idmc export --model Wan-AI/Wan2.1-T2V-1.3B --output ./wan_onnx \
                --extra-kwargs '{"guidance_scale": 5.0, "negative_prompt": "blurry"}'
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str):
    import torch
    mapping = {"float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float32": torch.float32, "fp32": torch.float32}
    if dtype_str not in mapping:
        raise argparse.ArgumentTypeError(
            f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping)}"
        )
    return mapping[dtype_str]


def _is_diffusion_model(model_id: str) -> bool:
    """Heuristic: check the model index to decide pipeline type."""
    try:
        from huggingface_hub import hf_hub_download
        import json as _json
        path = hf_hub_download(model_id, "model_index.json")
        idx = _json.loads(Path(path).read_text())
        return "_class_name" in idx and "Pipeline" in idx.get("_class_name", "")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Export command
# ---------------------------------------------------------------------------

def cmd_export(args):
    import torch

    torch_dtype = args.dtype

    # Build inference kwargs for diffusion models
    inf_kwargs = {}
    if args.prompt is not None:
        inf_kwargs["prompt"] = args.prompt
    if args.negative_prompt is not None:
        inf_kwargs["negative_prompt"] = args.negative_prompt
    if args.height is not None:
        inf_kwargs["height"] = args.height
    if args.width is not None:
        inf_kwargs["width"] = args.width
    if args.num_frames is not None:
        inf_kwargs["num_frames"] = args.num_frames
    if args.guidance_scale is not None:
        inf_kwargs["guidance_scale"] = args.guidance_scale
    inf_kwargs["num_inference_steps"] = args.num_inference_steps
    # Merge any extra kwargs
    if args.extra_kwargs:
        inf_kwargs.update(args.extra_kwargs)

    output = Path(args.output)

    # -----------------------------------------------------------------------
    # Diffusion pipeline export
    # -----------------------------------------------------------------------
    if args.task == "diffusion" or _is_diffusion_model(args.model):
        from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
            ORTDiffusionPipeline,
        )

        fixed_axes = args.fixed_axes or {}

        print(f"Exporting diffusion pipeline '{args.model}' → {output}")
        print(f"  provider : {args.provider}")
        print(f"  dtype    : {torch_dtype}")
        print(f"  inf_kwargs: {inf_kwargs}")

        pipe = ORTDiffusionPipeline.from_pretrained(
            args.model,
            provider=args.provider,
            torch_dtype=torch_dtype,
            export_by_inference=True,
            output=str(output),
            inference_kwargs=inf_kwargs,
            module_fixed_axis_fields=fixed_axes if fixed_axes else None,
        )
        print(f"\nExport complete. ORT pipeline loaded on: {pipe.device}")
        print(f"ONNX models saved to: {output.resolve()}")

    # -----------------------------------------------------------------------
    # Transformer / encoder-only export
    # -----------------------------------------------------------------------
    else:
        from inference_driven_model_compiler.optimum.onnxruntime.modeling_decoder import (
            OnTheFlyORTModelForCausalLM,
        )

        # Build encoder inference kwargs
        enc_kwargs: dict = {}
        if args.seq_length is not None:
            enc_kwargs["max_length"] = args.seq_length
        enc_kwargs.update(args.extra_kwargs or {})

        print(f"Exporting transformer model '{args.model}' → {output}")
        print(f"  provider  : {args.provider}")
        print(f"  dtype     : {torch_dtype}")
        print(f"  enc_kwargs: {enc_kwargs}")

        model = OnTheFlyORTModelForCausalLM.from_pretrained(
            args.model,
            provider=args.provider,
            torch_dtype=torch_dtype,
            export_by_inference=True,
            output=str(output),
            inference_kwargs=enc_kwargs if enc_kwargs else None,
        )
        print(f"\nExport complete. ONNX model saved to: {output.resolve()}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idmc",
        description="Inference-Driven Model Compiler — export any HuggingFace model to ONNX.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- export subcommand --------------------------------------------------
    exp = sub.add_parser(
        "export",
        help="Export a model to ONNX via inference-driven tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Required
    exp.add_argument("--model", required=True,
                     help="HuggingFace model ID or local path.")
    exp.add_argument("--output", required=True,
                     help="Directory to write ONNX models into.")

    # Pipeline control
    exp.add_argument("--task", default="auto",
                     choices=["auto", "diffusion", "text-generation",
                              "text-classification", "feature-extraction"],
                     help="Model task. 'auto' detects from model_index.json (default).")
    exp.add_argument("--provider", default="CUDAExecutionProvider",
                     help="ORT execution provider (default: CUDAExecutionProvider).")
    exp.add_argument("--dtype", default="float16", type=_parse_dtype,
                     metavar="{float16,bfloat16,float32}",
                     help="Torch dtype for the model (default: float16).")

    # Diffusion-specific kwargs
    diff = exp.add_argument_group("diffusion pipeline kwargs")
    diff.add_argument("--prompt", default=None,
                      help="Text prompt for the diffusion inference pass.")
    diff.add_argument("--negative-prompt", default=None,
                      help="Negative prompt.")
    diff.add_argument("--height", type=int, default=None,
                      help="Output video/image height in pixels.")
    diff.add_argument("--width", type=int, default=None,
                      help="Output video/image width in pixels.")
    diff.add_argument("--num-frames", type=int, default=None,
                      help="Number of video frames.")
    diff.add_argument("--guidance-scale", type=float, default=None,
                      help="Classifier-free guidance scale.")
    diff.add_argument("--num-inference-steps", type=int, default=1,
                      help="Denoising steps for shape capture (default: 1, keeps export fast).")

    # Transformer-specific kwargs
    enc = exp.add_argument_group("transformer / encoder kwargs")
    enc.add_argument("--seq-length", type=int, default=None,
                     help="Sequence length for encoder models.")

    # Advanced
    adv = exp.add_argument_group("advanced")
    adv.add_argument(
        "--fixed-axes",
        type=json.loads,
        default=None,
        metavar="JSON",
        help=(
            'JSON dict of {module: [field, ...]} whose config dimensions should be '
            'kept static. E.g. \'{"transformer": ["hidden_size"]}\''
        ),
    )
    adv.add_argument(
        "--extra-kwargs",
        type=json.loads,
        default=None,
        metavar="JSON",
        help="Extra pipeline/model kwargs as a JSON object.",
    )

    exp.set_defaults(func=cmd_export)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.error("Export failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
