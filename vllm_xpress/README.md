# xPress Serving in vLLM

Everything needed to run **xPress** (K-pass parallel Jacobi refinement over the
DFlash block drafter) as a first-class vLLM V1 speculative-decoding method.
Instead of vendoring the full vLLM tree, this folder ships a single patch
against upstream vLLM plus a one-shot setup script.

```
setup_vllm.sh                one command: clone upstream vLLM, install xPress, pip install
src/vllm/...                 the xPress source files (READ THEM HERE -- real code, not a diff):
  model_executor/models/xpress_head.py     the refiner head (pure torch, no vLLM imports)
  model_executor/models/qwen3_xpress.py    Qwen3XPressForCausalLM
  v1/worker/gpu/spec_decode/xpress/        speculator + Triton kernels
registration.patch           the ONLY edits to existing upstream files (~48 lines across 5
                             files: method registration in config/speculative.py,
                             config/vllm.py, models/registry.py, model_runner.py,
                             spec_decode/__init__.py). No other method is touched.
verify_install.py            post-install self-check: files landed, imports resolve, and
                             EVERY registration edit took effect (setup_vllm.sh runs it)
bench_vllm_accept.py         acceptance + throughput on gsm8k/math500/... (same protocol
                             as the HF harness in the parent folder)
convert_xpress_to_vllm.py    training checkpoint (.pt) -> vLLM serving directory
convert_markov_to_vllm_dspark.py  Markov training ckpt -> DSpark serving directory
```

Everything needed to reproduce the vLLM results lives in this folder: the source
is browsable above, and `setup_vllm.sh` reconstructs the exact tree the numbers
were measured on (upstream @ 82ae4164 + these files + registration.patch).

What the patch adds inside vLLM:

| Path (inside vllm) | Role |
|---|---|
| `vllm/model_executor/models/xpress_head.py` | pure-torch refiner head (importable without vLLM for CPU parity tests; mixer folded at load) |
| `vllm/model_executor/models/qwen3_xpress.py` | `Qwen3XPressForCausalLM` (shares target embed/lm_head; DFlash 5-layer backbone) |
| `vllm/v1/worker/gpu/spec_decode/xpress/` | `XPressSpeculator`: K loop-free Jacobi passes, CUDA-graph captured |
| small edits | method registration in `config/speculative.py`, `config/vllm.py`, `models/registry.py`, `model_runner.py`, `spec_decode/__init__.py` |

## 1. Setup (one command)

```bash
conda create -n vllm-xpress python=3.11 -y && conda activate vllm-xpress
./setup_vllm.sh            # clones upstream vllm @ 82ae4164, applies the patch, pip installs
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

`setup_vllm.sh` ends by running `verify_install.py`, which fails loudly if any
file or registration edit is missing. Re-run it any time:
`python verify_install.py`. At runtime the engine log must show both
`Asynchronous scheduling is enabled.` and `XPress: K=<n> Jacobi passes`.

(Equivalent manual steps: `git clone https://github.com/vllm-project/vllm.git && cd vllm &&
git checkout 82ae4164ee016d4daecd2033c26f5c0827984a80 && cp -r ../src/vllm/. vllm/ &&
git apply ../registration.patch && VLLM_USE_PRECOMPILED=1 pip install -e .`)

## 2. Checkpoint

The serving-format checkpoint (`Qwen3XPressForCausalLM`: `config.json` +
`model.safetensors`) lives in the release repo and is passed **by name**. vLLM
downloads it automatically.

```
UIUC-SSAIL/Qwen3-8B-XPress-b16
```

config.json knobs: `xpress_num_passes: 6` (ALWAYS check before benchmarking. It is a run knob living in config.json); `xpress_candidate_topc` must be **0 or absent**.

To convert a training checkpoint (`refiner_cotrain*.pt`) yourself:

```bash
python convert_xpress_to_vllm.py \
  --ckpt refiner_cotrain.pt \
  --base-config <local snapshot of z-lab/Qwen3-8B-DFlash-b16> \
  --out ckpts/Qwen3-8B-XPress-b16 --num-passes 6
```

## 3. Benchmark

`bench_vllm_accept.py` runs the SAME datasets and protocol as the HF harness in the parent folder (shuffle seed=0 subset, identical prompts, greedy, bs=1) and reads acceptance length from vLLM's spec-decode counters (AL = 1 + accepted/drafts, same per-step scale as the HF tau at block 16):

```bash
# xPress (checkpoint defaults to UIUC-SSAIL/Qwen3-8B-XPress-b16):
python bench_vllm_accept.py --dataset gsm8k --max-samples 128
# Markov-head baseline (vLLM's dspark method; ckpt defaults to UIUC-SSAIL/Qwen3-8B-Markov-b16):
python bench_vllm_accept.py --method dspark --dataset gsm8k --max-samples 128
# bare dFlash drafter baseline:
python bench_vllm_accept.py --method dflash --dataset gsm8k --max-samples 128
# AR baseline for the speedup denominator:
python bench_vllm_accept.py --no-spec --dataset gsm8k --max-samples 128
# other datasets: math500 humaneval mbpp aime25 mt-bench

# Sweeps run in ONE process (vLLM startup/teardown is slow -- 30+ CUDA graphs to
# capture and destroy per process, so never pay it per setting):
python bench_vllm_accept.py --dataset gsm8k,math500,humaneval --batch-size 1,4,8,16 --max-samples 128
# K (Jacobi passes) needs a fresh process -- it is read at speculator init:
for K in 4 5 6 7; do python bench_vllm_accept.py --num-passes $K --max-samples 128; done
```

Compare vLLM numbers against vLLM numbers (verify kernels/scheduler differ from the HF harness, so small AL offsets vs the HF table are expected).

**Large batch**: pass `--batch-size N` to submit N prompts per `generate()` call (vLLM's continuous batching schedules them together). Run the same sweep with `--no-spec` to find the spec-decode/AR crossover -- speculative decoding wins at low concurrency and loses at large batch by design (the verify passes compete with batch parallelism for the same FLOPs), so report the crossover batch, not a single number:

```bash
for B in 1 4 8 16 32; do
  python bench_vllm_accept.py --dataset gsm8k --max-samples 128 --batch-size $B
  python bench_vllm_accept.py --dataset gsm8k --max-samples 128 --batch-size $B --no-spec
done
```

For vLLM's STANDARD online-serving benchmark (request rates, TTFT/TPOT), serve the model and use `vllm bench serve` instead. This script covers offline acceptance + throughput on the release protocol.
