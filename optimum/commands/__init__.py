"""Local override of optimum.commands.

Extends __path__ to include the real site-packages optimum/commands directory so
that:
  * submodules we don't override (export/, env, base, optimum_cli, …) keep
    resolving to the real optimum/optimum-onnx package, and
  * our local ``register/`` directory is added to the
    ``optimum.commands.register`` PEP 420 namespace, letting optimum-cli
    auto-discover ``register_idmc.py`` and wire in the inference-driven flags.

This only takes effect when this repo is on sys.path *before* site-packages
(i.e. when the shadow ``optimum`` package is active). A plain optimum-cli with
this repo off the path is completely unaffected.
"""
from __future__ import annotations

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))

for _sp in sys.path:
    if not _sp or "inference_driven_model_compiler" in _sp:
        continue
    _real_dir = os.path.join(_sp, "optimum", "commands")
    if os.path.isdir(_real_dir) and _real_dir != _this_dir:
        # Merge: local overrides first, real package second.
        __path__ = [_this_dir, _real_dir]
        break
