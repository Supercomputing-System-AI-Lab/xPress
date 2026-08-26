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

<p align="center">
  <img src="images/speedup_t0_t1.png" alt="End-to-end speedup over autoregressive decoding on Qwen3-8B" width="100%">
</p>

<p align="center">
<em>End-to-end decoding speedup over autoregressive baseline (Qwen3-8B, block 16, batch 1).
xPress improves on the dFlash drafter at both temperatures on every benchmark, and the gains
survive lossless sampling — at T=1 it reaches 7.2× on GSM8K where the drafter alone gets 4.5×.</em>
</p>

---

## 🔍 How it works

A block-diffusion drafter emits a whole block in one pass, so its tokens are **per-position
marginals**: nothing ties token *k* to what the drafter itself just chose at *k-1*. The target
model verifies conditionally, so jointly-improbable blocks get rejected early.

xPress adds a small **causal refiner** on top of the drafter (left). The refiner turns the
drafter's logits into a corrected block by conditioning each position on the previous block
token, and the block is resolved with **K parallel Jacobi passes** instead of a left-to-right
loop — so the serial depth is K (~4-8), not the block size.

Everything but the two vocabulary projections lives in an `r`-dimensional bottleneck
(`r = 256`, right), which makes the head roughly 16× lighter than working at the model's
hidden size:

<p align="center">
  <img src="images/arch_paper.png" alt="xPress architecture" width="88%">
</p>

<p align="center">
<em>(a) the diffusion drafter with the causal refiner in the loop — the refiner reads the
drafter's hidden states and adds a logit bias on top of the target LM head's base logits.
(b) inside the refiner: previous-token embedding, per-position and block-global hidden states
are fused to <code>r</code>, mixed causally across the block, passed through an
<code>r</code>-space SwiGLU MLP, and read out to the vocabulary as a bias.</em>
</p>

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
conda create -n xpress python=3.11 -y
conda init bash && exec $SHELL      # skip if `conda activate` already works in your shell
conda activate xpress
pip install -r requirements.txt
```

> **Notes**
> - All reported acceptance lengths and throughputs were measured on a single **NVIDIA H200**. Verified with `torch==2.11.0+cu129`. Any CUDA 12.x-capable node works; if your driver is older than the wheel's CUDA, prepend the compat layer, e.g. `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`.
> - `flash-attn` is OPTIONAL: the harness auto-falls back to torch SDPA (published numbers were measured with SDPA).
> - First run compiles Triton/inductor kernels (~10 min); later runs reuse the cache (`./inductor_cache`, override with `INDUCTOR_CACHE=...`).

### 📦 Checkpoints

Checkpoints are passed **by Hugging Face repo id** and download automatically — the scripts already default to:

| checkpoint | contents |
|---|---|
| [`UIUC-SSAIL/Qwen3-8B-XPress-b16`](https://huggingface.co/UIUC-SSAIL/Qwen3-8B-XPress-b16) | xPress head + its co-trained drafter (and the vLLM serving format) |
| [`UIUC-SSAIL/Qwen3-8B-Markov-b16`](https://huggingface.co/UIUC-SSAIL/Qwen3-8B-Markov-b16) | Markov-head baseline + its co-trained drafter |

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

### 📊 Results

The figure at the top of this page reports **end-to-end speedup**; the table below reports the
**acceptance length** that produces it (mean accepted tokens per verification step, T=0).

**Every number here uses K = 6 Jacobi passes**, fixed across all benchmarks rather than tuned per
benchmark. `bench_fair_fast.sh` sweeps `K_LIST=4,5,6,7` and prints a row per K; take the K=6 row to
compare against this table (`K_LIST=6 ./bench_fair_fast.sh <dataset>` runs that single setting).

| benchmark | drafter | Markov | xPress |
|---|:---:|:---:|:---:|
| GSM8K | 6.48 | 9.67 | **10.11** |
| MATH500 | 7.71 | 9.24 | **9.62** |
| HumanEval | 6.44 | 7.76 | **8.15** |
| MBPP | 5.75 | 6.90 | **7.11** |
| AIME25<sup>†</sup> | 7.10 | 7.95 | **8.35** |
| LiveCodeBench | 7.11 | 7.85 | **8.40** |
| MT-Bench | 3.18 | 4.13 | **4.38** |

<sup>†</sup> AIME25 contains only **30 problems**, so `--max-samples 128` yields all 30 — this row
rests on far fewer samples than the others and its run-to-run spread is correspondingly wider.

Tolerances: acceptance ±0.1 (bf16 varies slightly across GPU models/driver stacks). Absolute tok/s depends on your node; the **paired ratio** printed by the interleaved script is the number to compare.

**On the choice of K.** K = 6 is a fixed, conservative setting, not a per-benchmark optimum:
acceptance is still rising at K = 7 on HumanEval and MT-Bench, and is non-monotonic on
LiveCodeBench (8.42 / 8.24 / 8.27 / 8.51 for K = 4/5/6/7), so no single K "saturates" everywhere.
Reporting one fixed K keeps the comparison against the drafter and the Markov head honest — tuning
K per benchmark would only raise the xPress column. Larger K costs one extra refiner pass each
(~85 µs at batch 1), which is why the throughput-optimal K can be smaller than the acceptance-optimal
one; sweep it for your own deployment.

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
