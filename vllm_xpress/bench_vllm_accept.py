import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
# the setup script may clone the vLLM SOURCE tree next to this script; drop the script's own
# dir from sys.path so `import vllm` resolves to the INSTALLED package, not the checkout
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
sys.path.insert(0, os.path.dirname(_here))

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--method", default="xpress", choices=["xpress", "dspark", "dflash"],
                    help="speculative method: xpress (our refiner), dspark (the Markov-head "
                    "baseline), dflash (bare drafter)")
parser.add_argument("--draft-model", default=None,
                    help="serving-format draft dir or HF repo id; defaults per --method: "
                    "xpress -> UIUC-SSAIL/Qwen3-8B-XPress-b16, dspark -> UIUC-SSAIL/Qwen3-8B-Markov-b16, "
                    "dflash -> z-lab/Qwen3-8B-DFlash-b16")
parser.add_argument("--dataset", default="gsm8k",
                    help="one name, or a comma list to sweep in ONE process "
                    "(gsm8k,math500,humaneval,...)")
parser.add_argument("--max-samples", type=int, default=128)
parser.add_argument("--max-new-tokens", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--num-speculative-tokens", type=int, default=15, help="drafts per step (block 16 = 15 + anchor)")
parser.add_argument("--no-spec", action="store_true", help="plain AR baseline (no speculative decoding)")
parser.add_argument("--max-model-len", type=int, default=8192)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
parser.add_argument("--num-passes", type=int, default=None,
                    help="xpress only: K Jacobi passes, overriding the checkpoint's "
                    "xpress_num_passes (released numbers use K=6)")
parser.add_argument("--batch-size", default="1",
                    help="prompts per generate() call; comma list sweeps in ONE process. "
                    "1 = protocol-matched sequential mode. N>1 uses vLLM continuous "
                    "batching -- sweep it to find the spec-decode/AR crossover.")
args = parser.parse_args()
if args.draft_model is None:
    args.draft_model = {"xpress": "UIUC-SSAIL/Qwen3-8B-XPress-b16",
                        "dspark": "UIUC-SSAIL/Qwen3-8B-Markov-b16",
                        "dflash": "z-lab/Qwen3-8B-DFlash-b16"}[args.method]

if args.num_passes is not None:
    os.environ["XPRESS_NUM_PASSES"] = str(args.num_passes)   # read by XPressSpeculator

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

datasets_to_run = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
batches_to_run = [int(b) for b in str(args.batch_size).split(",") if b.strip()]

tokenizer = AutoTokenizer.from_pretrained(args.model)


def build_prompts(name):
    ds = load_and_process_dataset(name)
    if args.max_samples is not None and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=0).select(range(args.max_samples))
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": inst["turns"][0]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for inst in ds
    ]


prompt_sets = {d: build_prompts(d) for d in datasets_to_run}
for d, ps in prompt_sets.items():
    print(f"[bench] {d}: {len(ps)} prompts (turn 1 only; shuffle seed=0 subset)")

kwargs = dict(
    model=args.model,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    disable_log_stats=False,
)
if not args.no_spec:
    kwargs["speculative_config"] = {
        "model": args.draft_model,
        "method": args.method,
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
llm.generate([prompt_sets[datasets_to_run[0]][0]],
             SamplingParams(temperature=args.temperature, max_tokens=64))

results = []
for ds_name in datasets_to_run:
    prompts = prompt_sets[ds_name]
    for B in batches_to_run:
        c0 = spec_counters()
        new_tokens, decode_time = 0, 0.0
        t_all = time.perf_counter()
        for i0 in range(0, len(prompts), B):
            chunk = prompts[i0:i0 + B]
            t0 = time.perf_counter()
            outs = llm.generate(chunk, sp)
            dt = time.perf_counter() - t0
            new_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
            decode_time += dt
        wall = time.perf_counter() - t_all
        c1 = spec_counters()

        def delta(frag):
            for k in c1:
                if frag in k and isinstance(c1[k], (int, float)):
                    return c1[k] - c0.get(k, 0)
            return None

        tput = new_tokens / decode_time
        al = None
        if not args.no_spec:
            drafts, accepted = delta("num_drafts"), delta("num_accepted_tokens")
            if drafts and accepted is not None:
                al = 1.0 + accepted / drafts
        results.append((ds_name, B, new_tokens, wall, tput, al))
        tag = "AR baseline" if args.no_spec else args.method
        kk = "" if args.num_passes is None else f" K={args.num_passes}"
        print(f"\n=== {ds_name}  batch={B}  T={args.temperature}  {tag}{kk}")
        print(f"    tokens={new_tokens}  wall={wall:.1f}s  POOLED={tput:.1f} tok/s"
              + (f"  AL={al:.2f}" if al else ""))

print("\n" + "=" * 72)
print(f"SUMMARY  {'AR baseline' if args.no_spec else args.method}"
      f"{'' if args.num_passes is None else f' K={args.num_passes}'}  N={args.max_samples}")
print(f"{'dataset':<14}{'batch':>7}{'tok/s':>12}{'AL':>8}")
for ds_name, B, ntok, wall, tput, al in results:
    print(f"{ds_name:<14}{B:>7}{tput:>12.1f}{(f'{al:.2f}' if al else '-'):>8}")
print("=" * 72)
