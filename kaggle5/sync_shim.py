#!/usr/bin/env python3
"""Bake lever3_runtime.py into lever3_apply.py as base64.

The GPU kernel gets only the patch cell and the applier cell, and it has no
internet. The shim therefore has to travel inside the applier. Run this after
editing lever3_runtime.py; build_notebook.py fails loudly if you forget.
"""
import base64
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / "lever3_runtime.py"
APPLIER = HERE / "lever3_apply.py"

raw = SHIM.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
b64 = base64.b64encode(raw).decode()

text = APPLIER.read_text()
text, n1 = re.subn(r'SHIM_SHA256 = "[0-9a-f]*"  # k5-shim-sha',
                   'SHIM_SHA256 = "%s"  # k5-shim-sha' % digest, text)
text, n2 = re.subn(r'SHIM_B64 = "[A-Za-z0-9+/=]*"  # k5-shim-b64',
                   'SHIM_B64 = "%s"  # k5-shim-b64' % b64, text)
if n1 != 1 or n2 != 1:
    raise SystemExit("could not find the shim markers in lever3_apply.py")
APPLIER.write_text(text)
print("shim baked: %d bytes -> %d base64 chars, sha256 %s"
      % (len(raw), len(b64), digest[:16]))
