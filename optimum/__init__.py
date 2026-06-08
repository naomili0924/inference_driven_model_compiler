"""
Local optimum package root.

Extends __path__ to include the real site-packages optimum/ directory so that
all submodules (commands, exporters, onnxruntime, …) remain accessible even
when this directory is listed on sys.path before site-packages.  Our local
overrides (e.g. optimum/exporters/onnx/) take precedence because they appear
first in __path__.
"""
from __future__ import annotations
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))

for _sp in sys.path:
    if not _sp or "inference_driven_model_compiler" in _sp:
        continue
    _real_dir = os.path.join(_sp, "optimum")
    if os.path.isdir(_real_dir) and _real_dir != _this_dir:
        # Merge: local overrides first, real package second
        __path__ = [_this_dir, _real_dir]
        break
