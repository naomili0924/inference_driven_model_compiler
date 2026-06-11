"""Auto-registration hook that adds inference-driven export to ``optimum-cli``.

optimum-cli discovers every ``*.py`` file under the ``optimum.commands.register``
PEP 420 namespace at startup and reads each one's ``REGISTER_COMMANDS`` list.
This repo ships its copy of that namespace (via the shadow ``optimum.commands``
package), so this file is imported whenever the shadow ``optimum`` is active.

Rather than register a *second* ``export onnx`` subcommand (which would collide
with the one optimum-onnx registers), we patch the existing ``ONNXExportCommand``
in place so that ``optimum-cli export onnx`` gains three extra flags:

    --export_by_inference       enable inference-driven tracing
    --inference_kwargs          JSON dict of inputs used to trace the model
    --module_fixed_axis_fields  JSON dict of {module: [config_field, ...]} static dims

and routes them to the inference-driven ``main_export`` (which the shadow
``optimum.exporters.onnx.__main__`` already provides). This mirrors the
``IDMCONNXExportCommand`` defined in ``inference_driven_model_compiler/cli.py``,
but is kept self-contained here so it works with only the repo dir on PYTHONPATH
(no dependency on the ``inference_driven_model_compiler`` package being importable).

Usage (no launcher needed)::

    PYTHONPATH=/path/to/inference_driven_model_compiler \
        optimum-cli export onnx --model <id> <outdir> \
            --task text-generation-with-past --dtype fp16 --device cpu \
            --export_by_inference=true
"""
from __future__ import annotations

import json
import os


def _parse_bool_arg(val):
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes")


def _json_or_file(val):
    """Parse a JSON CLI argument from either an inline string or a file.

    Accepts, in order:
      * ``@/path/to/file.json`` — explicit file reference (curl-style)
      * an existing file path (e.g. ``inputs.json``) — read as JSON
      * an inline JSON string (e.g. ``'{"input_ids": []}'``)

    This lets large inputs (e.g. a multimodal ``inference_kwargs`` with a big
    ``pixel_values`` array) live in a file instead of a giant command-line string.
    """
    if isinstance(val, str) and val.startswith("@"):
        with open(os.path.expanduser(val[1:])) as f:
            return json.load(f)
    if isinstance(val, str) and os.path.isfile(val):
        with open(val) as f:
            return json.load(f)
    return json.loads(val)


def _patch_onnx_export_command() -> None:
    from optimum.commands.export import onnx as _onnx_module

    command_cls = _onnx_module.ONNXExportCommand
    if getattr(command_cls, "_idmc_registered", False):
        return

    @staticmethod
    def parse_args(parser):
        # All the standard onnx export flags first ...
        from optimum.commands.export.onnx import parse_args_onnx

        parse_args_onnx(parser)

        # ... then the inference-driven additions.
        group = parser.add_argument_group("Inference-driven export (IDMC)")
        group.add_argument(
            "--inference_kwargs",
            type=_json_or_file,
            default=None,
            metavar="JSON|FILE",
            help=(
                "JSON dict of model inputs for tracing, given inline or as a file "
                "path (e.g. '@inputs.json'). An empty list [] means auto-generate a "
                'dummy tensor for that input. Example: \'{"input_ids": []}\' or @inputs.json'
            ),
        )
        group.add_argument(
            "--module_fixed_axis_fields",
            type=_json_or_file,
            default=None,
            metavar="JSON|FILE",
            help=(
                "JSON dict of {module_name: [config_field, ...]} whose config dimensions "
                "should be treated as static (non-dynamic) axes, given inline or as a file "
                'path. Example: \'{"transformer": ["hidden_size"]}\' or @axes.json'
            ),
        )
        group.add_argument(
            "--export_by_inference",
            type=_parse_bool_arg,
            default=False,
            metavar="BOOL",
            help="Enable inference-driven export (default: false).",
        )
        group.add_argument(
            "--fixed_inputs",
            type=_json_or_file,
            default=None,
            metavar="JSON|FILE",
            help=(
                "JSON list of input names whose traced tensor VALUES must be replayed "
                "verbatim during export/validation instead of being randomly regenerated "
                "from their shape, given inline or as a file path. Use for inputs whose "
                "values control graph structure (sizes/indices) or are coupled to other "
                'inputs, where a random value would break the trace. '
                'Example: \'["image_grid_thw", "cache_position"]\' or @fixed.json'
            ),
        )

    def run(self):
        from optimum.exporters.onnx.__main__ import main_export
        from optimum.utils.input_generators import DEFAULT_DUMMY_SHAPES

        # Build input shapes from the standard dummy-shape args.
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES:
            if hasattr(self.args, input_name):
                input_shapes[input_name] = getattr(self.args, input_name)

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
            fixed_inputs=self.args.fixed_inputs,
            **input_shapes,
        )

    command_cls.parse_args = parse_args
    command_cls.run = run
    command_cls._idmc_registered = True


# Patch on import (optimum-cli imports this module during command discovery,
# before the onnx subparser is built and before any command is run).
_patch_onnx_export_command()

# We don't add a new command — we patched the existing one in place.
REGISTER_COMMANDS: list = []
