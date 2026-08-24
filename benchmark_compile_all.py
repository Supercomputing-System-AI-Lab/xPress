import argparse
import time
import random
from itertools import chain
from loguru import logger
import numpy as np
import torch
import torch._dynamo
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from model import load_and_process_dataset
import distributed as dist
from bench_utils import (load_draft_model_for_benchmark, spec_gen,
                         warmup_pipeline, report_compile_cache, setup_compile_all)
from bench_metrics import write_answers, summarize, agg, print_latency_profile

import os


def _run_timed_eval(dataset, tokenizer, target, draft_model, args, block_size, bs_list, runtime):
    answers = []
    indices = range(dist.rank(), len(dataset), dist.size())
    _timed_ic = torch.inference_mode()
    _timed_ic.__enter__()
    try:
        for idx in tqdm(indices, disable=not dist.is_main()):
            instance = dataset[idx]
            messages = []
            choice_b1 = {"index": 0, "block_size": 1, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
            choice_bk = {"index": 1, "block_size": block_size, "turns": [], "new_tokens": [], "wall_time": [], "prefill_times": [], "decode_times": [], "acceptance_lengths": []}
            for turn_index, user_content in enumerate(instance["turns"]):
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

                response = {}
                for bs in bs_list:
                    response[bs] = spec_gen(draft_model, target, tokenizer, args,
                                             input_ids, args.max_new_tokens, bs, runtime)

                # Record results for the batch sizes actually run (see bs_list)
                for bs in bs_list:
                    choice = choice_b1 if bs == 1 else choice_bk
                    r = response[bs]
                    generated_ids = r.output_ids[0, r.num_input_tokens:]
                    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    choice["turns"].append(output_text)
                    choice["new_tokens"].append(int(r.num_output_tokens))
                    prefill_t = float(r.time_to_first_token)
                    decode_t = float(r.time_per_output_token) * int(r.num_output_tokens)
                    choice["prefill_times"].append(prefill_t)
                    choice["decode_times"].append(decode_t)
                    choice["wall_time"].append(prefill_t + decode_t)
                    choice["acceptance_lengths"].append([int(x) for x in r.acceptance_lengths])

                # Use b=k result as conversation history (same as original logic); ar-only falls back to b=1
                spec_response = response[1 if args.ar_only else block_size]
                generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})

            answers.append({
                "question_id": idx,
                "choices": [choice_b1, choice_bk],
                "tstamp": time.time(),
            })
    finally:
        _timed_ic.__exit__(None, None, None)

    if dist.size() > 1:
        answers = dist.gather(answers, dst=0)
        if not dist.is_main():
            return None
        answers = list(chain(*answers))

    answers.sort(key=lambda x: x["question_id"])
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--draft-name-or-path", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-temperature", type=float, default=None,
                        help="Temperature for the DRAFT proposals (default: same as --temperature, "
                        "the DeepSpec convention). 0 = greedy drafting; verification then uses the "
                        "exact match against a sampled target posterior (accept prob = p(proposed "
                        "token) -- distributionally identical to one-hot rejection). Lossless either way.")
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--answer-file", type=str, default=None, help="Output answer file (jsonl) to store generation results for both b=1 and b=k.")
    parser.add_argument("--attn-implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation for target and draft models. Default: auto-detect flash_attn.")
    parser.add_argument("--xpress-refiner-path", type=str, default=None,
                        help="refiner_cotrain.pt for the XPress refiner head; also overrides the "
                        "drafter with the checkpoint's co-trained draft_state_dict. Head config is "
                        "rebuilt from ckpt['args'].")
    parser.add_argument("--markov-refiner-path", type=str, default=None,
                        help="refiner_cotrain.pt for a MARKOV-ONLY head. Loads the FAITHFUL DeepSpec "
                        "VanillaMarkov (sequential sample_block_tokens); our w1/w2 remapped to markov_w1/w2. "
                        "Sequential (no K); ignores --xpress-refine-passes. Mutually exclusive with the others.")
    parser.add_argument("--profile-latency", action="store_true",
                        help="Per-block LATENCY PROFILE of the draft-side stages (drafter forward / base "
                        "lm_head readout / refiner correction), measured on REAL inputs during the run and "
                        "summarised at the end. Works for the XPress and Markov refiners, so it is the "
                        "way to compare the XPress refiner vs the Markov head head-to-head. NOTE: it inserts a "
                        "cuda.synchronize per stage per block -> the reported THROUGHPUT is perturbed; use a "
                        "separate un-profiled run for throughput numbers.")
    parser.add_argument("--xpress-refine-passes", type=int, default=3, help="K parallel XPress refine passes.")
    parser.add_argument("--compile-refiner", action="store_true",
                        help="torch.compile the refiner (fuse its small kernels) before graph capture.")
    parser.add_argument("--drafter-only", action="store_true",
                        help="BASELINE ablation: plain DFlash drafter, NO refiner (forces K=0 -> draft = "
                        "the drafter's parallel-argmax seed). Without a refiner path -> vanilla drafter; "
                        "WITH --xpress-refiner-path -> the co-trained drafter alone (eager)." )
    parser.add_argument("--compile-all", action="store_true",
                        help="ABSOLUTE-THROUGHPUT mode: torch.compile the WHOLE pipeline -- target (verify "
                        "AND b=1 AR baseline, so the speedup ratio stays fair), drafter forward, and refiner. "
                        "Uses dynamic=True for the growing KV cache. First sample(s) are slow (compilation).")
    parser.add_argument("--ar-only", action="store_true",
                        help="Run ONLY the b=1 pure-target AR baseline (skip the b=k spec path). Isolates the "
                        "AR baseline -- fast way to sanity-check it / measure how much --compile-all speeds up "
                        "the plain target forward. Reports b1_AR only (no speedup/accept).")
    parser.add_argument("--skip-ar", action="store_true",
                        help="Skip the b=1 AR baseline (it is slow + stable); run ONLY the b=k spec path. "
                        "Reports bK_spec / spec_throughput / accept. Pass --ar-baseline-ms to also get the "
                        "speedup ratio. NOTE: loses the per-run b1_AR throttle tell -- watch the node.")
    parser.add_argument("--ar-baseline-ms", type=float, default=None,
                        help="With --skip-ar: reference AR ms/tok (e.g. the value you measured once with "
                        "--ar-only) to print speedup = ar_baseline_ms / bK_spec without re-running AR.")
    parser.add_argument("--inductor-cache-dir", type=str,
                        default="./inductor_cache",
                        help="Persistent torch.compile/inductor cache dir (shared FS) so the ~10min first "
                        "compile is reused by later runs. Only used with --compile-all.")
    parser.add_argument("--fair-interleave", type=int, default=0, metavar="ROUNDS",
                        help="FAIR PAIRED BENCHMARK in ONE process: give BOTH --xpress-refiner-path and "
                        "--markov-refiner-path and run them alternately for ROUNDS rounds. The target/drafter "
                        "are loaded and compiled ONCE and shared, and the per-method co-trained drafter "
                        "weights are swapped in with load_state_dict (guards key on shape/dtype/device, NOT "
                        "on values -> no recompile). Kills the ~2min dynamo re-trace + model load that every "
                        "separate process pays. Round order is ALTERNATED (A,B / B,A / ...) so the second-mover "
                        "warm-cache advantage cancels. Prints the per-round paired ratios + aggregate.")
    parser.add_argument("--no-fold-mixer", action="store_true",
                        help="Disable the inference-only mixer fold (L <- L*tril + I, which drops the mask "
                        "multiply and the residual add from every Jacobi pass). The fold is algebraically "
                        "exact, so this flag exists to A/B its LATENCY on one node -- and as an escape hatch "
                        "if acceptance ever moves (it must not).")
    parser.add_argument("--fair-k-list", type=str, default=None, metavar="4,5,6",
                        help="With --fair-interleave: sweep these K values for XPress in the SAME process, "
                        "all against ONE shared Markov reference (the Markov head decodes sequentially and "
                        "ignores K, so it needs no variant). K does NOT trigger a recompile -- it is a Python "
                        "loop over a compiled head called with identical shapes, so only a cheap CUDA-graph "
                        "re-capture per K. Default: just --xpress-refine-passes.")
    parser.add_argument("--fair-warmup-rounds", type=int, default=1,
                        help="With --fair-interleave: discard the first N rounds (cold caches / GPU not yet at "
                        "boost). Order-based exclusion declared up front -- not cherry-picking. Still printed.")

    args = parser.parse_args()

    if args.model_name_or_path is None or args.draft_name_or_path is None:
        parser.error("--model-name-or-path and --draft-name-or-path are required")
    if args.ar_only and args.skip_ar:
        parser.error("--ar-only and --skip-ar are mutually exclusive")
    if args.drafter_only and args.fair_interleave > 0:
        parser.error("--drafter-only is a single-method baseline; do not combine with --fair-interleave")
    if args.fair_interleave > 0 and args.fair_k_list:
        _ks = [int(x) for x in args.fair_k_list.split(",") if x.strip()]
        if not _ks:
            parser.error("--fair-k-list is empty")
        if any(k <= 0 for k in _ks):
            parser.error(f"--fair-k-list needs K >= 1 (use --drafter-only for K=0), got {_ks}")
        args.xpress_refine_passes = _ks[0]

    if args.profile_latency:
        os.environ["TIME_DRAFTER"] = "1"
        os.environ["TIME_BASE"] = "1"
        os.environ["TIME_REFINER"] = "1"

    _seed = int(os.environ.get("DFLASH_SEED", "0"))
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    if args.attn_implementation is not None:
        attn_impl = args.attn_implementation
        logger.info(f"Using specified attention implementation: {attn_impl}")
    else:
        def has_flash_attn():
            if is_flash_attn_2_available():
                return True
            logger.warning("FlashAttention2 is not available. Falling back to torch.sdpa. The speedup will be lower.")
            return False

        installed_flash_attn = has_flash_attn()
        attn_impl = "flash_attention_2" if installed_flash_attn else "sdpa"
        logger.info(f"Auto-detected attention implementation: {attn_impl}")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = load_draft_model_for_benchmark(
        args.draft_name_or_path,
        attn_impl,
    ).to(device).eval()
    logger.info(f"[VERIFY] Target attn_implementation: {target.config._attn_implementation}")
    logger.info(f"[VERIFY] Draft attn_implementation: {draft_model.config._attn_implementation}")

    if args.compile_all:
        target = setup_compile_all(target, draft_model, args, dist.is_main())
        args.compile_refiner = True

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    bs_list = [1] if args.ar_only else ([block_size] if args.skip_ar else [1, block_size])
    cotrain_sd = {}

    xpress_refiner = None
    if args.xpress_refiner_path is not None:
        _ck = torch.load(args.xpress_refiner_path, map_location="cpu", weights_only=False)
        assert "draft_state_dict" in _ck, (
            "--xpress-refiner-path must be a CO-TRAIN checkpoint (with draft_state_dict). "
            "Use --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16."
        )
        _miss, _unexp = draft_model.load_state_dict(_ck["draft_state_dict"], strict=False)
        if dist.is_main():
            print(f"[xpress] checkpoint step={_ck.get('global_step')}")
            print(f"[xpress] overrode drafter with CO-TRAINED weights: missing={len(_miss)} unexpected={len(_unexp)}")
            if _miss:
                print(f"[xpress] WARNING drafter keys NOT loaded (stay at base z-lab): {list(_miss)[:8]}")
        if args.fair_interleave > 0:
            cotrain_sd["xpress"] = _ck["draft_state_dict"]
        del _ck
        from refiners.xpress import load_xpress_refiner
        xpress_refiner = load_xpress_refiner(
            args.xpress_refiner_path,
            vocab_size=draft_model.config.vocab_size,
            hidden_size=draft_model.config.hidden_size,
            block_size=block_size,
            device=device,
            dtype=torch.bfloat16,
            fold_mixer=not args.no_fold_mixer,
        )
        if args.compile_refiner:
            xpress_refiner = torch.compile(xpress_refiner)
            if dist.is_main():
                print("[xpress] torch.compile'd head (kernel fusion; first block slower for warmup).")

    markov_refiner = None
    if args.markov_refiner_path is not None:
        assert args.xpress_refiner_path is None or args.fair_interleave > 0, \
            "--markov-refiner-path and --xpress-refiner-path can only coexist under --fair-interleave."
        _ck = torch.load(args.markov_refiner_path, map_location="cpu", weights_only=False)
        assert "draft_state_dict" in _ck, (
            "--markov-refiner-path must be a CO-TRAIN checkpoint (with draft_state_dict). "
            "Use --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16."
        )
        _miss, _unexp = draft_model.load_state_dict(_ck["draft_state_dict"], strict=False)
        if dist.is_main():
            print(f"[markov] checkpoint step={_ck.get('global_step')}")
            print(f"[markov] overrode drafter with CO-TRAINED weights: missing={len(_miss)} unexpected={len(_unexp)}")
            if _miss:
                print(f"[markov] WARNING drafter keys NOT loaded (stay at base z-lab): {list(_miss)[:8]}")
        if args.fair_interleave > 0:
            cotrain_sd["markov"] = _ck["draft_state_dict"]
        del _ck
        from refiners.markov import load_markov_head
        markov_refiner = load_markov_head(
            args.markov_refiner_path, vocab_size=draft_model.config.vocab_size, device=device, dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    hidden_size = int(target.lm_head.weight.shape[1])
    vocab_size = int(target.lm_head.weight.shape[0])
    xpress_graph_runner = None
    markov_graph_runner = None
    drafter_seed_graph_runner = None
    if args.drafter_only:
        args.xpress_refine_passes = 0
        if args.use_graph:
            from refiners.xpress import DrafterSeedGraphRunner
            _draft_t = args.temperature if args.draft_temperature is None else args.draft_temperature
            drafter_seed_graph_runner = DrafterSeedGraphRunner(
                block_size=block_size, vocab_size=vocab_size, device=device, temperature=_draft_t)
            if dist.is_main():
                _mode = "argmax" if _draft_t == 0 else f"frozen-Gumbel T={_draft_t}"
                print(f"[drafter-only] BASELINE: plain drafter, K=0. Seed sampling CUDA-graphed ({_mode}).")
        elif dist.is_main():
            print(f"[drafter-only] BASELINE: plain drafter, NO refiner (K=0). Eager path.")
        args.use_graph = False

    if args.use_graph and xpress_refiner is None and markov_refiner is None:
        raise RuntimeError(
            "--use-graph needs a refiner: pass --xpress-refiner-path (XPress) or "
            "--markov-refiner-path; for the bare-drafter baseline pass --drafter-only "
            "(its K=0 seed sampling is CUDA-graphed, so --use-graph is fine there too)."
        )
    elif args.use_graph and xpress_refiner is not None:
        from refiners.xpress import XPressRefineGraphRunner
        _draft_t = args.temperature if args.draft_temperature is None else args.draft_temperature
        _lock = args.temperature > 0
        xpress_graph_runner = XPressRefineGraphRunner(
            head=xpress_refiner, block_size=block_size,
            refine_passes=args.xpress_refine_passes, hidden_dim=hidden_size,
            vocab_size=vocab_size, device=device,
            temperature=_draft_t, lock=_lock,
        )
        if dist.is_main():
            _mode = "greedy" if _draft_t == 0 else f"frozen-Gumbel T={_draft_t}{' +lock' if _lock else ''}"
            print(f"[xpress] CUDA-graph par-K rollout ENABLED (K={args.xpress_refine_passes}, {_mode}).")
    elif args.use_graph and markov_refiner is not None:
        from refiners.markov import MarkovSeqGraphRunner
        _draft_t = args.temperature if args.draft_temperature is None else args.draft_temperature
        markov_graph_runner = MarkovSeqGraphRunner(
            markov_head=markov_refiner, block_size=block_size,
            vocab_size=vocab_size, device=device, temperature=_draft_t,
        )
        if dist.is_main():
            _mode = "greedy" if _draft_t == 0 else f"sampling T={_draft_t} (in-graph multinomial)"
            print(f"[markov] CUDA-graph SEQUENTIAL decode ENABLED (block={block_size}, {_mode}).")

    if (args.fair_interleave > 0 and args.use_graph and markov_refiner is not None
            and markov_graph_runner is None):
        from refiners.markov import MarkovSeqGraphRunner
        markov_graph_runner = MarkovSeqGraphRunner(
            markov_head=markov_refiner, block_size=block_size,
            vocab_size=vocab_size, device=device,
            temperature=(args.temperature if args.draft_temperature is None
                         else args.draft_temperature),
        )
        if dist.is_main():
            print(f"[fair] Markov CUDA-graph runner built alongside the XPress one (block={block_size}).")

    runtime = dict(
        xpress_refine_passes=args.xpress_refine_passes,
        xpress_refiner=xpress_refiner,
        markov_refiner=markov_refiner,
        xpress_graph_runner=xpress_graph_runner,
        markov_graph_runner=markov_graph_runner,
        drafter_seed_graph_runner=drafter_seed_graph_runner,
        drafter_only=args.drafter_only,
    )

    if args.fair_interleave > 0:
        assert xpress_refiner is not None and markov_refiner is not None, \
            "--fair-interleave needs BOTH --xpress-refiner-path and --markov-refiner-path."
        assert "xpress" in cotrain_sd and "markov" in cotrain_sd, \
            "--fair-interleave needs CO-TRAIN checkpoints (with draft_state_dict) for both methods."
        assert not getattr(torch._inductor.config, "freezing", False), \
            "torch._inductor.config.freezing=True constant-folds weights -> the drafter swap would be a no-op."
            
        k_list = ([int(x) for x in args.fair_k_list.split(",") if x.strip()]
                  if args.fair_k_list else [args.xpress_refine_passes])
        assert k_list, "--fair-k-list parsed empty"


        hyb_runners = {}
        if args.use_graph:
            from refiners.xpress import XPressRefineGraphRunner
            _draft_t = args.temperature if args.draft_temperature is None else args.draft_temperature
            _lock = args.temperature > 0
            for _K in k_list:
                if xpress_graph_runner is not None and _K == args.xpress_refine_passes:
                    hyb_runners[_K] = xpress_graph_runner        # reuse the one already built above
                else:
                    hyb_runners[_K] = XPressRefineGraphRunner(
                        head=xpress_refiner, block_size=block_size,
                        refine_passes=_K, hidden_dim=hidden_size, vocab_size=vocab_size, device=device,
                        temperature=_draft_t, lock=_lock)
            if dist.is_main():
                print(f"[fair] XPress CUDA-graph runners captured for K={k_list} (no recompile: same shapes)")

        def _rt_xpress(K):
            rt = dict(runtime)
            rt["xpress_refiner"]      = xpress_refiner
            rt["xpress_graph_runner"] = hyb_runners.get(K)
            rt["xpress_refine_passes"]   = K 
            rt["markov_refiner"]      = None
            rt["markov_graph_runner"] = None
            return rt

        def _rt_markov():
            rt = dict(runtime)
            rt["xpress_refiner"]      = None
            rt["xpress_graph_runner"] = None
            rt["markov_refiner"]      = markov_refiner
            rt["markov_graph_runner"] = markov_graph_runner
            return rt

        specs = [(f"K{K}", f"XPress(K={K})", "xpress", _rt_xpress(K)) for K in k_list]
        specs.append(("markov", "Markov", "markov", _rt_markov()))
        REF = "markov"

        def _activate(sd_key):
            draft_model.load_state_dict(cotrain_sd[sd_key], strict=False)

        if args.compile_all:
            _t_warm = time.time()
            for key, name, sd_key, rt in specs:
                _activate(sd_key)
                if dist.is_main():
                    print(f"[fair] warming up {name} (untimed)...", flush=True)
                warmup_pipeline(dataset, tokenizer, target, draft_model, args, bs_list, rt)
            if dist.is_main():
                report_compile_cache(time.time() - _t_warm)
                print(f"[fair] warmup done for all {len(specs)} variants; timing starts now.", flush=True)

        tput = {s[0]: [] for s in specs}
        ratios = {s[0]: [] for s in specs}
        tputP = {s[0]: [] for s in specs}
        ratiosP = {s[0]: [] for s in specs}
        accept = {}
        taus = {s[0]: {"step": [], "sample": []} for s in specs}
        for rnd in range(1, args.fair_interleave + 1):
            # ROTATE the order every round (counterbalancing): with >2 variants a plain A/B swap no longer
            # balances, but a cyclic rotation gives every variant each slot equally often over a full cycle.
            shift = (rnd - 1) % len(specs)
            order = specs[shift:] + specs[:shift]
            is_warm = rnd <= args.fair_warmup_rounds
            res = {}
            for key, name, sd_key, rt in order:
                _activate(sd_key)
                if dist.is_main():
                    print(f"\n>>> round {rnd}/{args.fair_interleave} : {name}"
                          f"{'   [warmup - EXCLUDED]' if is_warm else ''}", flush=True)
                ans = _run_timed_eval(dataset, tokenizer, target, draft_model, args,
                                      block_size, bs_list, rt)
                if ans is None: 
                    continue
                if args.answer_file and rnd == args.fair_interleave:
                    _root, _ext = os.path.splitext(args.answer_file)
                    write_answers(ans, f"{_root}.{key}{_ext or '.jsonl'}")
                res[key] = summarize(ans, args, block_size, verbose=False)
                accept.setdefault(key, res[key])
                if args.temperature > 0 and not is_warm:
                    taus[key]["step"].append(res[key]["tau_step"])
                    taus[key]["sample"].append(res[key]["tau_sample"])
            if dist.is_main() and not is_warm and len(res) == len(specs):
                for key, name, _s, _r in specs:
                    tput[key].append(res[key]["throughput"])
                    ratios[key].append(res[key]["throughput"] / res[REF]["throughput"])
                    tputP[key].append(res[key]["throughput_pooled"])
                    ratiosP[key].append(res[key]["throughput_pooled"] / res[REF]["throughput_pooled"])
                print("    round %d (pooled): %s" % (rnd, "   ".join(
                    f"{name} {res[key]['throughput_pooled']:.1f}" for key, name, _s, _r in specs)))

        if dist.is_main() and ratios[REF]:
            n = len(ratios[REF])
            print("\n" + "=" * 96)
            print(f"FAIR INTERLEAVED RESULT   dataset={args.dataset}   block={block_size}   "
                  f"rounds={args.fair_interleave} (first {args.fair_warmup_rounds} discarded -> N={n})")
            print(f"{'method':<16}{'POOLED tok/s':>17}{'ratio':>16}{'tau/step':>10}"
                  f"{'per-sample tok/s':>19}{'ratio':>16}{'tau/sample':>12}")
            for key, name, _s, _r in specs:
                SP, RP = agg(tputP[key]), agg(ratiosP[key], "%.4f")
                SS, RS = agg(tput[key]),  agg(ratios[key],  "%.4f")
                cp = "1.0000 (ref)" if key == REF else f"{RP['mean']}+-{RP['std']}"
                cs = "1.0000 (ref)" if key == REF else f"{RS['mean']}+-{RS['std']}"
                _tst = (sum(taus[key]["step"]) / len(taus[key]["step"])
                        if taus[key]["step"] else accept[key]["tau_step"])
                _tsa = (sum(taus[key]["sample"]) / len(taus[key]["sample"])
                        if taus[key]["sample"] else accept[key]["tau_sample"])
                print(f"{name:<16}{SP['mean'] + '+-' + SP['std']:>17}{cp:>16}"
                      f"{_tst:>10.2f}"
                      f"{SS['mean'] + '+-' + SS['std']:>19}{cs:>16}"
                      f"{_tsa:>12.2f}")
            print("-" * 96)
            print("SELF-CONSISTENT PAIRS: (POOLED tok/s, tau/step) and (per-sample tok/s, tau/sample).")
            print("Never quote tau/step next to the per-sample throughput -- they weight samples differently")
            print("(pooling weights by token count, the per-sample metric weights every prompt equally), so")
            print("the pair will look inconsistent wherever the gain is concentrated in long generations.")
            print("-" * 96)
            for key, name, _s, _r in specs:
                if key == REF:
                    continue
                RP = agg(ratiosP[key], "%.4f")
                RS = agg(ratios[key], "%.4f")
                print(f"{name:<16} POOLED ratios: {' '.join('%.4f' % r for r in ratiosP[key])}"
                      f"  -> {(RP['_mean'] - 1) * 100:+.1f} +- {float(RP['std']) * 100:.1f} % "
                      f"(median {(RP['_median'] - 1) * 100:+.1f}%)"
                      f"   | per-sample -> {(RS['_mean'] - 1) * 100:+.1f} +- {float(RS['std']) * 100:.1f} %"
                      f" (median {(RS['_median'] - 1) * 100:+.1f}%)")
            if args.temperature > 0:
                print("accept is STOCHASTIC (T>0): tau above = mean over the measured rounds.")
            else:
                print("accept is deterministic (greedy): identical every round, no repeats needed.")
            print("NOTE: absolute tok/s carries one-sided cluster noise (clocks not lockable) and is biased")
            print("      LOW by any statistic; the PAIRED RATIOS are the numbers to report -- within a round")
            print("      every variant shares the same thermal/allocator state, so that drift cancels.")
            print("NOTE: pick K by where ACCEPT saturates, not by the throughput peak -- picking the argmax")
            print("      of a noisy throughput curve selects noise and overstates the winner.")
            print("=" * 96)
        return

    if args.compile_all:
        if dist.is_main():
            print("[compile-all] warming up compilation (untimed)...", flush=True)
        _t_warm = time.time()
        warmup_pipeline(dataset, tokenizer, target, draft_model, args, bs_list, runtime)
        if dist.is_main():
            report_compile_cache(time.time() - _t_warm)
            print("[compile-all] warmup done; timing starts now.", flush=True)

    answers = _run_timed_eval(dataset, tokenizer, target, draft_model, args,
                              block_size, bs_list, runtime)
    if answers is None: 
        return

    if args.answer_file and dist.is_main():
        write_answers(answers, args.answer_file)

    summarize(answers, args, block_size)

    if args.profile_latency:
        print_latency_profile(xpress_refiner, markov_refiner)


if __name__ == "__main__":
    main()
