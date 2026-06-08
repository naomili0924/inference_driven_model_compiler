from __future__ import annotations
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))

for _sp in sys.path:
    if not _sp or "inference_driven_model_compiler" in _sp:
        continue
    _real_dir = os.path.join(_sp, "optimum", "exporters")
    if os.path.isdir(_real_dir) and _real_dir != _this_dir:
        __path__ = [_this_dir, _real_dir]
        break
