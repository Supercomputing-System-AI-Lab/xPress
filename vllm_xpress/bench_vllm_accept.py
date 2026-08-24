import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--draft-model", default=None, help="serving-format XPress dir (config.json + model.safetensors)")
parser.add_argument("--dataset", default="gsm8k")
parser.add_argument("--max-samples", type=int, default=128)
parser.add_argument("--max-new-tokens", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--num-speculative-tokens", type=int, default=15, help="drafts per step (block 16 = 15 + anchor)")
parser.add_argument("--no-spec", action="store_true", help="plain AR baseline (no speculative decoding)")
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
args = parser.parse_args()
if not args.no_spec and args.draft_model is None:
    parser.error("--draft-model is required unless --no-spec")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from model.utils import load_and_process_dataset          # release-root layout
except ModuleNotFoundError:
    # standalone fallback: same datasets + prompt templates as model/utils.py
    from datasets import load_dataset

    def load_and_process_dataset(name):
        box = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        if name == "gsm8k":
            d = load_dataset("openai/gsm8k", "main", split="test")
            return d.map(lambda x: {"turns": [("{question}" + box).format(**x)]})
        if name == "math500":
            d = load_dataset("HuggingFaceH4/MATH-500", split="test")
            return d.map(lambda x: {"turns": [("{problem}" + box).format(**x)]})
        if name == "aime25":
            d = load_dataset("MathArena/aime_2025", split="train")
            return d.map(lambda x: {"turns": [("{problem}" + box).format(**x)]})
        if name == "humaneval":
            d = load_dataset("openai/openai_humaneval", split="test")
            fmt = ("Write a solution to the following problem and make sure that it passes "
                   "the tests:\n```python\n{prompt}\n```")
            return d.map(lambda x: {"turns": [fmt.format(**x)]})
        if name == "mbpp":
            d = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            return d.map(lambda x: {"turns": [x["prompt"]]})
        if name == "mt-bench":
            d = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            return d.map(lambda x: {"turns": x["prompt"]})
        raise ValueError(f"standalone fallback supports gsm8k/math500/aime25/humaneval/mbpp; "
                         f"run from the release root for {name!r}")

dataset = load_and_process_dataset(args.dataset)
if args.max_samples is not None and len(dataset) > args.max_samples:
    dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

tokenizer = AutoTokenizer.from_pretrained(args.model)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": inst["turns"][0]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    for inst in dataset
]
print(f"[bench] {args.dataset}: {len(prompts)} prompts (turn 1 only; shuffle seed=0 subset)")

kwargs = dict(
    model=args.model,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    disable_log_stats=False,
)
if not args.no_spec:
    kwargs["speculative_config"] = {
        "model": args.draft_model,
        "method": "xpress",
        "num_speculative_tokens": args.num_speculative_tokens,
    }
llm = LLM(**kwargs)
sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)


def spec_counters():
    vals = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                vals[m.name] = getattr(m, "value", None) or getattr(m, "values", None)
    except Exception as e:
        print(f"[bench] get_metrics unavailable ({e}); acceptance will not be reported")
    return vals


# untimed warmup (compiles / captures graphs)
llm.generate([prompts[0]], SamplingParams(temperature=args.temperature, max_tokens=64))

c0 = spec_counters()
new_tokens, decode_time = 0, 0.0
t_all = time.perf_counter()
for i, p in enumerate(prompts):
    t0 = time.perf_counter()
    out = llm.generate([p], sp)
    dt = time.perf_counter() - t0
    n = len(out[0].outputs[0].token_ids)
    new_tokens += n
    decode_time += dt
    if (i + 1) % 16 == 0:
        print(f"[bench] {i + 1}/{len(prompts)}  running POOLED throughput ~ {new_tokens / decode_time:.1f} tok/s")
wall = time.perf_counter() - t_all
c1 = spec_counters()

print("\n" + "=" * 72)
print(f"RESULT  dataset={args.dataset}  N={len(prompts)}  T={args.temperature}  "
      f"{'AR baseline (no spec)' if args.no_spec else f'XPress K(config) block={args.num_speculative_tokens + 1}'}")
print(f"tokens = {new_tokens}   wall = {wall:.1f}s")
print(f"POOLED throughput = {new_tokens / decode_time:.1f} tok/s   (sum tokens / sum per-request time, bs=1)")
if not args.no_spec:
    def delta(name_frag):
        for k in c1:
            if name_frag in k and isinstance(c1[k], (int, float)):
                return c1[k] - c0.get(k, 0)
        return None
    drafts = delta("num_drafts")
    accepted = delta("num_accepted_tokens")
    if drafts and accepted is not None:
        print(f"drafts = {drafts:.0f}   accepted = {accepted:.0f}")
        print(f"ACCEPTANCE LENGTH (per step) = 1 + accepted/drafts = {1.0 + accepted / drafts:.2f}")
    else:
        print("spec-decode counters not found; available:", sorted(c1) or "(none)")
print("=" * 72)
