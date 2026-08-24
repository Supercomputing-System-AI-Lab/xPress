# XPress Serving in vLLM

Everything needed to run **XPress** (K-pass parallel Jacobi refinement over the
DFlash block drafter) as a first-class vLLM V1 speculative-decoding method.
Instead of vendoring the full vLLM tree, this folder ships a single patch
against upstream vLLM; one `git apply` reproduces our serving branch.

```
xpress_vllm.patch            all XPress changes vs upstream vllm @ 82ae4164
                             (12 files, ~940 added lines; no other methods touched)
specdec_bench_harness.patch  adds XPRESS to NVIDIA SpecDec-Bench (37 lines)
```

What the patch adds:

| Path (inside vllm) | Role |
|---|---|
| `vllm/model_executor/models/xpress_head.py` | pure-torch refiner head (importable without vLLM for CPU parity tests; mixer folded at load) |
| `vllm/model_executor/models/qwen3_xpress.py` | `Qwen3XPressForCausalLM` (shares target embed/lm_head; DFlash 5-layer backbone) |
| `vllm/v1/worker/gpu/spec_decode/xpress/` | `XPressSpeculator`: K loop-free Jacobi passes, CUDA-graph captured; optional fused Triton kernels (off by default) |
| small edits | method registration in `config/speculative.py`, `config/vllm.py`, `models/registry.py`, `model_runner.py`, `spec_decode/__init__.py` |
| `docs/xpress/README.md` | the full serving/benchmark guide (this file's long form) |

Also in this folder: `convert_xpress_to_vllm.py` — converts a training
checkpoint (`refiner_cotrain*.pt`) into the vLLM serving directory.

## Setup

```bash
conda create -n vllm-xpress python=3.11 -y && conda activate vllm-xpress
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout 82ae4164ee016d4daecd2033c26f5c0827984a80
git apply /path/to/xpress_eval_release/vllm_xpress/xpress_vllm.patch
# Python-only changes -> use the prebuilt wheel for the base commit:
VLLM_USE_PRECOMPILED=1 pip install -e .
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

## Checkpoint (serving format)

```bash
hf download VictorZheng/Qwen3-8B-XPress-b16-consis-k3 --local-dir ckpts/Qwen3-8B-XPress-b16
```

The serving format is `config.json` + `model.safetensors` (co-trained drafter
weights + head under the `xpress_head.` prefix, `Qwen3XPressModel` architecture).
If the repo only has the training checkpoint (`refiner_cotrain*.pt`), convert it
yourself with the bundled script:

```bash
python convert_xpress_to_vllm.py \
  --ckpt refiner_cotrain.pt \
  --base-config <local snapshot of z-lab/Qwen3-8B-DFlash-b16> \
  --out ckpts/Qwen3-8B-XPress-b16 \
  --num-passes 6
```

config.json knobs: `xpress_num_passes: 6` (ALWAYS check before benchmarking —
it is a run knob living in config.json); `xpress_candidate_topc` must be **0 or
absent** (full-vocab, exact semantics — the release default);
`xpress_fused_kernel: false`.

## Benchmark A: standard datasets (gsm8k / math500 / ...)

`bench_vllm_accept.py` runs the SAME datasets and protocol as the HF harness in
the parent folder (shuffle seed=0 subset, identical prompts, greedy, bs=1) and
reads acceptance length from vLLM's spec-decode counters
(AL = 1 + accepted/drafts, same per-step scale as the HF tau at block 16).
Run it FROM THE RELEASE ROOT (it imports the shared dataset loaders):

```bash
python vllm_xpress/bench_vllm_accept.py \
  --draft-model ckpts/Qwen3-8B-XPress-b16 --dataset gsm8k --max-samples 128
# AR baseline for the speedup denominator:
python vllm_xpress/bench_vllm_accept.py --no-spec --dataset gsm8k --max-samples 128
```

Compare vLLM numbers against vLLM numbers (verify kernels/scheduler differ from
the HF harness, so small AL offsets vs the HF table are expected).

## Benchmark B: NVIDIA SpecDec-Bench

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer && cd TensorRT-Model-Optimizer
git apply /path/to/xpress_eval_release/vllm_xpress/specdec_bench_harness.patch
cd examples/specdec_bench
python run.py \
  --model_dir Qwen/Qwen3-8B --tokenizer Qwen/Qwen3-8B \
  --draft_model_dir /path/to/ckpts/Qwen3-8B-XPress-b16 \
  --mtbench question.jsonl --engine VLLM --speculative_algorithm XPRESS \
  --block_size 15 --tp_size 1 --ep_size 1 --output_length 2048 \
  --concurrency 1 --show_progress
```

Expected (Qwen3-8B, H200, T=0): MT-Bench bs=1 AL ~3.84, ~560 tok/s
(bare drafter 2.98 / 455).

NOTE: the acceptance-length convention here (SpecDec-Bench, block 15) differs
from the HF harness in the parent folder (block 16, per-step tau over the full
block); compare vLLM numbers only against vLLM numbers.
