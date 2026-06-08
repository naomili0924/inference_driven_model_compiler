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

This module also provides IDMCONNXExportCommand, registered with optimum-cli as
a replacement for the built-in "export onnx" command so that the new
inference-driven flags are available:

    optimum-cli export onnx \
        --model sentence-transformers/paraphrase-MiniLM-L12-v2 \
        /targetdir/paraphrase-MiniLM-L12-v2 \
        --inference_kwargs='{"input_ids": []}' \
        --module_fixed_axis_fields='{"transformer": ["hidden_size"]}' \
        --export_by_inference=true
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import ArgumentParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (idmc CLI)
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
    try:
        from huggingface_hub import hf_hub_download
        import json as _json
        path = hf_hub_download(model_id, "model_index.json")
        idx = _json.loads(Path(path).read_text())
        return "_class_name" in idx and "Pipeline" in idx.get("_class_name", "")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# idmc export command
# ---------------------------------------------------------------------------

def cmd_export(args):
    import torch

    torch_dtype = args.dtype

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
    if args.extra_kwargs:
        inf_kwargs.update(args.extra_kwargs)

    output = Path(args.output)

    if args.task == "diffusion" or _is_diffusion_model(args.model):
        from inference_driven_model_compiler.optimum.onnxruntime.modeling_diffusion import (
            ORTDiffusionPipeline,
        )

        fixed_axes = args.fixed_axes or {}

        print(f"Exporting diffusion pipeline '{args.model}' → {output}")
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
    else:
        from inference_driven_model_compiler.optimum.onnxruntime.modeling_decoder import (
            OnTheFlyORTModelForCausalLM,
        )

        enc_kwargs: dict = {}
        if args.seq_length is not None:
            enc_kwargs["max_length"] = args.seq_length
        enc_kwargs.update(args.extra_kwargs or {})

        print(f"Exporting transformer model '{args.model}' → {output}")
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
# idmc argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idmc",
        description="Inference-Driven Model Compiler — export any HuggingFace model to ONNX.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser(
        "export",
        help="Export a model to ONNX via inference-driven tracing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    exp.add_argument("--model", required=True,
                     help="HuggingFace model ID or local path.")
    exp.add_argument("--output", required=True,
                     help="Directory to write ONNX models into.")
    exp.add_argument("--task", default="auto",
                     choices=["auto", "diffusion", "text-generation",
                              "text-classification", "feature-extraction"],
                     help="Model task. 'auto' detects from model_index.json (default).")
    exp.add_argument("--provider", default="CUDAExecutionProvider",
                     help="ORT execution provider (default: CUDAExecutionProvider).")
    exp.add_argument("--dtype", default="float16", type=_parse_dtype,
                     metavar="{float16,bfloat16,float32}",
                     help="Torch dtype for the model (default: float16).")

    diff = exp.add_argument_group("diffusion pipeline kwargs")
    diff.add_argument("--prompt", default=None)
    diff.add_argument("--negative-prompt", default=None)
    diff.add_argument("--height", type=int, default=None)
    diff.add_argument("--width", type=int, default=None)
    diff.add_argument("--num-frames", type=int, default=None)
    diff.add_argument("--guidance-scale", type=float, default=None)
    diff.add_argument("--num-inference-steps", type=int, default=1)

    enc = exp.add_argument_group("transformer / encoder kwargs")
    enc.add_argument("--seq-length", type=int, default=None)

    adv = exp.add_argument_group("advanced")
    adv.add_argument("--fixed-axes", type=json.loads, default=None, metavar="JSON")
    adv.add_argument("--extra-kwargs", type=json.loads, default=None, metavar="JSON")

    exp.set_defaults(func=cmd_export)
    return parser


# ---------------------------------------------------------------------------
# optimum-cli integration: IDMCONNXExportCommand
# ---------------------------------------------------------------------------

def _parse_bool_arg(val: str) -> bool:
    if isinstance(val, bool):
        return val
    return val.lower() in ("1", "true", "yes")


class IDMCONNXExportCommand:
    """
    Replacement for optimum's built-in ONNXExportCommand.

    Adds three inference-driven flags on top of all the standard flags:
      --inference_kwargs       JSON dict, empty list means auto-generate tensor
      --module_fixed_axis_fields  JSON dict of {module: [config_field, ...]}
      --export_by_inference    bool flag to enable inference-driven tracing
    """

    COMMAND = None  # set after import of CommandInfo

    @staticmethod
    def _setup_command():
        from optimum.commands.base import BaseOptimumCLICommand, CommandInfo
        IDMCONNXExportCommand.__bases__ = (BaseOptimumCLICommand,)
        IDMCONNXExportCommand.COMMAND = CommandInfo(
            name="onnx",
            help="Export PyTorch to ONNX (inference-driven)",
        )

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        from optimum.commands.export.onnx import parse_args_onnx
        parse_args_onnx(parser)

        idmc_group = parser.add_argument_group("Inference-driven export (IDMC)")
        idmc_group.add_argument(
            "--inference_kwargs",
            type=json.loads,
            default=None,
            metavar="JSON",
            help=(
                "JSON dict of model inputs for tracing. "
                "An empty list [] means auto-generate a dummy tensor for that input. "
                "Example: '{\"input_ids\": [], \"attention_mask\": []}'"
            ),
        )
        idmc_group.add_argument(
            "--module_fixed_axis_fields",
            type=json.loads,
            default=None,
            metavar="JSON",
            help=(
                "JSON dict of {module_name: [config_field, ...]} whose config dimensions "
                "should be treated as static (non-dynamic) axes. "
                "Example: '{\"transformer\": [\"hidden_size\", \"vocab_size\"]}'"
            ),
        )
        idmc_group.add_argument(
            "--export_by_inference",
            type=_parse_bool_arg,
            default=False,
            metavar="BOOL",
            help="Enable inference-driven export (default: false).",
        )

    def run(self):
        from optimum.utils.input_generators import DEFAULT_DUMMY_SHAPES

        # Build input shapes from standard dummy-shape args
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES:
            if hasattr(self.args, input_name):
                input_shapes[input_name] = getattr(self.args, input_name)

        # Import our inference-driven main_export (not the stock optimum one)
        from optimum.exporters.onnx.__main__ import main_export

        main_export(
            model_name_or_path=self.args.model,
            output=self.args.output,
            task=self.args.task,
            opset=self.args.opset,
            device=self.args.device,
            dtype=self.args.dtype,
            optimize=self.args.optimize,
            monolith=self.args.monolith,
            no_post_process=self.args.no_post_process,
            framework=self.args.framework,
            atol=self.args.atol,
            cache_dir=self.args.cache_dir,
            trust_remote_code=self.args.trust_remote_code,
            pad_token_id=self.args.pad_token_id,
            use_subprocess=False,
            _variant=self.args.variant,
            library_name=self.args.library_name,
            no_dynamic_axes=self.args.no_dynamic_axes,
            model_kwargs=self.args.model_kwargs,
            do_constant_folding=not self.args.no_constant_folding,
            slim=self.args.slim,
            dynamo=getattr(self.args, "dynamo", False),
            inference_kwargs=self.args.inference_kwargs,
            module_fixed_axis_fields=self.args.module_fixed_axis_fields,
            export_by_inference=self.args.export_by_inference,
            **input_shapes,
        )


def _build_idmc_onnx_export_command():
    """Return IDMCONNXExportCommand as a proper BaseOptimumCLICommand subclass."""
    from optimum.commands.base import BaseOptimumCLICommand, CommandInfo

    class _IDMCONNXExportCommand(IDMCONNXExportCommand, BaseOptimumCLICommand):
        COMMAND = CommandInfo(
            name="onnx",
            help="Export PyTorch to ONNX (inference-driven)",
        )

    return _IDMCONNXExportCommand


# ---------------------------------------------------------------------------
# Entry point (idmc CLI)
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
