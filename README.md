<div align="center">
<h1>XPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding</h1>

<p align="center">
  <a href="https://github.com/Supercomputing-System-AI-Lab/xPress">
    <img src="https://img.shields.io/badge/%20GitHub-000000?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2608.02438">
    <img src="https://img.shields.io/badge/%20arXiv-CC0000?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
</p>

</div>

---

## 🧠 Abstract

Block-diffusion drafters like dFlash generate an entire block of draft tokens in a single forward pass, drastically reducing the overhead of multiple-token drafting in speculative decoding. The crucial final step of the single-pass discrete denoising process involves using the logit distribution at each position to sample conditionally independent tokens. The resulting draft is thus a set of per-position marginals, rather than a joint distribution: no draft token is guaranteed to depend on its predecessors. Such independently sampled marginals tend to produce sequences with tokens that are individually likely, but jointly improbable under the target model's distribution, which verifies each token conditionally. This can cause early rejection and limits acceptance length. To address this, we propose **xPress** as a means to restore the missing causality in diffusion drafters. xPress is a lightweight causal refiner that reconciles the whole diffusion block at once through parallel refinement, restoring and propagating causal dependencies across the draft without a token-by-token loop. On Qwen3-8B, across seven math, code, and chat benchmarks, xPress raises **acceptance length by ~30% on average (up to +56%)** and its end-to-end decoding throughput by **~1.3× on average (up to 1.7×)** compared to the original dFlash diffusion drafter.

---

## 📋 What's in this repo

The exact HF-based harness that produced the paper's acceptance-length and throughput numbers (Qwen3-8B target, dFlash-b16 drafter, block 16), plus a self-contained vLLM serving integration:

| Component | Path |
|---|---|
| Main eval harness (timed loop + fair-interleave protocol) | `benchmark_compile_all.py` |
| Model loading / warmup / torch.compile helpers | `bench_utils.py` |
| Throughput & acceptance metrics, latency profiles | `bench_metrics.py` |
| dFlash drafter + spec-decode loop (T=1 lossless verification) | `model/dflash.py` |
| Dataset loading + prompt templates (part of the protocol) | `model/utils.py` |
| **xPress refiner head** (self-contained torch) | `refiners/xpress_head.py` |
| xPress loader + par-K Jacobi rollout + CUDA-graph runner | `refiners/xpress.py` |
| Markov-head baseline (drives DeepSpec's own implementation) | `refiners/markov.py` |
| Vendored, unmodified DeepSpec code (see `PROVENANCE.md`) | `refiners/deepspec/` |
| Published protocol, T=0 and T=1 in one script | `bench_fair_fast.sh` |
| **vLLM serving**: xPress source + install script + benchmarks | `vllm_xpress/` |

---

## 🚀 Getting Started

### 🖥️ Environment Setup

```bash
conda create -n dflash python=3.11 -y
conda activate dflash
pip install -r requirements.txt
```

> **Notes**
> - Verified with `torch==2.11.0+cu129` on H200. Any CUDA 12.x-capable node works; if your driver is older than the wheel's CUDA, prepend the compat layer, e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`.
> - `flash-attn` is OPTIONAL: the harness auto-falls back to torch SDPA (published numbers were measured with SDPA).
> - First run compiles Triton/inductor kernels (~10 min); later runs reuse the cache (`./inductor_cache`, override with `INDUCTOR_CACHE=...`).

### 📦 Checkpoints

Checkpoints are passed **by Hugging Face repo id** and download automatically — the scripts already default to:

```
UIUC-SSAIL/Qwen3-8B-XPress-b16   # xPress head + co-trained drafter
UIUC-SSAIL/Qwen3-8B-Markov-b16   # Markov head + co-trained drafter
```

These repos are private during review: run `hf auth login` once with an account that has access. To use local files instead, point `XPRESS_CKPT` / `MK_CKPT` (or `--xpress-refiner-path` / `--markov-refiner-path`) at a `.pt` path. The base drafter (`z-lab/Qwen3-8B-DFlash-b16`) and target (`Qwen/Qwen3-8B`) are public and download automatically.

---

## ⚖️ Reproduce the paper numbers

One script for both temperatures — `./bench_fair_fast.sh <dataset> [temperature]`:

```bash
# T=0 acceptance + paired throughput (published protocol; interleaved rounds,
# warmup discarded, paired ratios):
./bench_fair_fast.sh gsm8k
# T=1 lossless sampling (frozen-Gumbel + honest-q verification):
./bench_fair_fast.sh gsm8k 1
# All benchmarks: gsm8k math500 humaneval mbpp aime25 livecodebench mt-bench
```

Bare-drafter baseline (the `drafter` column below): pass `--drafter-only` to `benchmark_compile_all.py` (draft = the drafter's own K=0 seed, no refiner; works with or without `--use-graph` — the seed sampling itself is CUDA-graphed).

Single-method / custom runs go through `benchmark_compile_all.py` directly (see `--help`); the `.sh` wrapper only sets the published protocol.

### 📊 Expected results (acceptance length per step, T=0)

| benchmark | drafter | Markov | xPress |
|---|:---:|:---:|:---:|
| GSM8K | 6.48 | 9.67 | **10.11** |
| MATH500 | 7.71 | 9.24 | **9.62** |
| HumanEval | 6.44 | 7.76 | **8.15** |
| MBPP | 5.75 | 6.90 | **7.11** |
| AIME25 | 7.10 | 7.95 | **8.35** |
| LiveCodeBench | 7.11 | 7.85 | **8.40** |
| MT-Bench | 3.18 | 4.13 | **4.38** |

Tolerances: acceptance ±0.1 (bf16 varies slightly across GPU models/driver stacks). Absolute tok/s depends on your node; the **paired ratio** printed by the interleaved script is the number to compare.

> **Protocol** (already baked into the scripts — do not change when reproducing): subset = `datasets.shuffle(seed=0)`, first 128 samples; temperature as named; `max_new_tokens=2048`; block 16; CUDA graph + torch.compile on. Multi-turn datasets (MT-Bench): all turns are generated (turn 2 conditions on turn 1's output) but the reported metrics cover **turn 1 only** — the numbers above use this convention. The interleave rotation is exactly balanced when the measured rounds are a multiple of the variant count; the script defaults (ROUNDS=6 at T=0, 5 at T=1, first round discarded) satisfy this.

---

## ⚡ Serving with vLLM

[`vllm_xpress/`](vllm_xpress/) runs xPress as a first-class vLLM V1 speculative-decoding method. Our implementation lives there as **readable source** (`vllm_xpress/src/vllm/...`: the refiner head, the draft model, the speculator and its Triton kernels); the only edits to existing upstream files are ~48 lines of method registration in [`registration.patch`](vllm_xpress/registration.patch). `setup_vllm.sh` clones upstream vLLM at a pinned commit, installs our files, and self-checks the result:

```bash
conda create -n vllm-xpress python=3.11 -y && conda activate vllm-xpress
cd vllm_xpress && ./setup_vllm.sh
python bench_vllm_accept.py --dataset gsm8k --max-samples 128           # xPress
python bench_vllm_accept.py --method dspark --dataset gsm8k --max-samples 128   # Markov baseline
```

See [`vllm_xpress/README.md`](vllm_xpress/README.md) for the serving checkpoint, the K/batch sweeps, and the runtime log lines that confirm the integration is active.

---

## 🙏 Attribution

The Markov-head **baseline** runs DeepSeek's own implementation, vendored verbatim under `refiners/deepspec/` — see [`refiners/deepspec/PROVENANCE.md`](refiners/deepspec/PROVENANCE.md). Everything under `refiners/xpress*` is ours.

---

## 📚 Citation

If you find our work useful or relevant to your project and research, please kindly cite:

```bibtex
@misc{wang2026xpress,
      title={xPress: Parallel Refinement for Diffusion Drafters in Speculative Decoding},
      author={Zheng Wang and Davis Wertheimer and Yu Chin Fabian Lim and Mudhakar Srivatsa and Raghu K. Ganti and Minjia Zhang and Naigang Wang},
      year={2026},
      eprint={2608.02438},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.02438},
}
```
