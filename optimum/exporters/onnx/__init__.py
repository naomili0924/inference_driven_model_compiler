"""
Local override of optimum.exporters.onnx.

Extends the real site-packages package by:
- Adding our local modules (utils.py, convert.py, model_configs.py, etc.) which
  take precedence for submodule lookups.
- Forwarding all attribute lookups for symbols we don't define (export_models,
  validate_models_outputs, OnnxConfig, …) to the real package.
- Overriding main_export with the inference-driven version from __main__.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys

# ── Extend __path__ to include the real site-packages directory ──────────────
# This lets Python find submodules we don't override (base.py, config.py, …)
# from the real optimum-onnx package while our local files take precedence.
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _sp in sys.path:
    if not _sp or "inference_driven_model_compiler" in _sp:
        continue
    _real_dir = os.path.join(_sp, "optimum", "exporters", "onnx")
    if os.path.isdir(_real_dir) and _real_dir != _this_dir:
        __path__ = [_this_dir, _real_dir]
        break


def __getattr__(name: str):
    """Lazily forward missing attributes to the real optimum.exporters.onnx.

    Submodules that exist as local .py files are imported via the extended __path__
    so our overrides take precedence over the real package's lazy-module mechanism.
    """
    import importlib as _importlib

    # If this attribute is a local submodule, import it via the full dotted name
    # so Python uses our extended __path__ (which puts local dir first).
    _local_py = os.path.join(_this_dir, name + ".py")
    _local_pkg = os.path.join(_this_dir, name, "__init__.py")
    if os.path.exists(_local_py) or os.path.exists(_local_pkg):
        return _importlib.import_module(f"optimum.exporters.onnx.{name}")

    # Fall through to the real package for everything else.
    _key = "_idmc_real_optimum_onnx_init"
    if _key not in sys.modules:
        for _p in sys.path:
            if not _p or "inference_driven_model_compiler" in _p:
                continue
            _f = os.path.join(_p, "optimum", "exporters", "onnx", "__init__.py")
            if os.path.exists(_f):
                spec = importlib.util.spec_from_file_location(_key, _f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[_key] = mod
                spec.loader.exec_module(mod)
                break
    real = sys.modules.get(_key)
    if real is not None:
        try:
            return getattr(real, name)
        except AttributeError:
            pass
    raise AttributeError(f"module 'optimum.exporters.onnx' has no attribute {name!r}")


def main_export(*args, **kwargs):
    """Inference-driven ONNX export — delegates to __main__.main_export."""
    from optimum.exporters.onnx.__main__ import main_export as _impl
    return _impl(*args, **kwargs)
