#!/usr/bin/env python3
"""Verify the XPress changes are present AND active in the installed vLLM.

Run after setup_vllm.sh (it calls this automatically). Checks the files landed,
the imports resolve, and every registration edit took effect -- including the
async-scheduling whitelist, which fails silently (vLLM just logs a warning and
runs ~17% slower) if it is missing.
"""
import pathlib
import sys

FAILURES = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {label}" + (f"   -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(label)


import vllm  # noqa: E402

root = pathlib.Path(vllm.__file__).parent
print(f"installed vLLM: {root}")

print("\n1. XPress source files present")
for rel in (
    "model_executor/models/xpress_head.py",
    "model_executor/models/qwen3_xpress.py",
    "v1/worker/gpu/spec_decode/xpress/__init__.py",
    "v1/worker/gpu/spec_decode/xpress/speculator.py",
    "v1/worker/gpu/spec_decode/xpress/kernels.py",
):
    check(rel, (root / rel).exists())

print("\n2. imports resolve")
try:
    from vllm.model_executor.models.xpress_head import XPressRefinerHead  # noqa: F401
    from vllm.model_executor.models.qwen3_xpress import (  # noqa: F401
        Qwen3XPressForCausalLM,
    )
    from vllm.v1.worker.gpu.spec_decode.xpress.speculator import (  # noqa: F401
        XPressSpeculator,
    )

    check("xpress head / model / speculator", True)
except Exception as e:  # pragma: no cover
    check("xpress head / model / speculator", False, repr(e))

print("\n3. registration edits applied to upstream files")
MARKERS = {
    "config/speculative.py": [("XPressModelTypes", "method type"),
                              ('"xpress"', "config branch")],
    "config/vllm.py": [('!= "xpress"', "async-scheduling whitelist"),
                       ('"xpress",', "supported-method list")],
    "model_executor/models/registry.py": [("Qwen3XPressModel", "architecture registry")],
    "v1/worker/gpu/model_runner.py": [('"xpress"', "aux hidden states")],
    "v1/worker/gpu/spec_decode/__init__.py": [("XPressSpeculator", "speculator dispatch")],
}
for rel, markers in MARKERS.items():
    text = (root / rel).read_text()
    for marker, what in markers:
        check(f"{rel}: {what}", marker in text, f"missing marker {marker!r}")

# the async-scheduling whitelist must appear TWICE (explicit-enable + auto-enable paths);
# only one of them means half the code paths still silently disable overlap.
n = (root / "config/vllm.py").read_text().count('!= "xpress"')
check("config/vllm.py: async-scheduling whitelist in BOTH paths", n == 2, f"found {n}, expected 2")

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    print("The install is incomplete -- re-run setup_vllm.sh in a clean directory.")
    sys.exit(1)
print("All XPress changes present and active.")
print("At runtime, confirm the engine log shows BOTH:")
print("  'Asynchronous scheduling is enabled.'      <- the +17% host/GPU overlap")
print("  'XPress: K=<n> Jacobi passes'              <- the head is driving the draft")
