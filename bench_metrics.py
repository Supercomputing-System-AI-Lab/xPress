import json
import os
from itertools import chain

import numpy as np
from rich import print


def write_answers(answers, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ans in answers:
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")
    print(f"Saved answer file with {len(answers)} samples to {path}")


def pooled_tps(answers, ci):
    """
    POOLED throughput = sum(tokens) / sum(decode time) over the whole set -- the literal wall-clock
    number ("run this workload, get this many tok/s"). Distinct from the per-sample metric below, which
    averages each sample's OWN rate and so weights a 50-token and a 2000-token generation equally.
    """
    tok = sum(a["choices"][ci]["new_tokens"][0] for a in answers if a["choices"][ci]["new_tokens"])
    tim = sum(a["choices"][ci]["decode_times"][0] for a in answers if a["choices"][ci]["new_tokens"])
    return (tok / tim) if tim > 0 else 0.0


def summarize(answers, args, block_size, verbose=True):
    """Print the throughput / acceptance stats and return them as a dict."""
    out = {}
    if args.ar_only:
        # AR-baseline isolation: only b=1 was run. Report the AR throughput; no speedup/accept.
        t1 = np.mean([ans["choices"][0]["decode_times"][0] / max(1, ans["choices"][0]["new_tokens"][0]) for ans in answers if ans["choices"][0]["new_tokens"]])
        pooled = pooled_tps(answers, 0)
        print(f"[AR-only] b1_AR = {t1 * 1000:.3f} ms/tok   AR_throughput = {1.0 / t1:.1f} tok/s "
              f"(per-sample)   {pooled:.1f} tok/s (POOLED)")
        return {"ms_per_tok": t1 * 1000, "throughput": 1.0 / t1, "throughput_pooled": pooled}

    tb = np.mean([ans["choices"][1]["decode_times"][0] / max(1, ans["choices"][1]["new_tokens"][0]) for ans in answers if ans["choices"][1]["new_tokens"]])
    out["ms_per_tok"] = tb * 1000
    out["throughput"] = 1.0 / tb
    out["throughput_pooled"] = pooled_tps(answers, 1)
    if args.skip_ar:
        # b=1 AR baseline skipped -> no t1. Report bK_spec/throughput; speedup only if --ar-baseline-ms given.
        _line = f"[spec-only] bK_spec = {tb * 1000:.3f} ms/tok   spec_throughput = {1.0 / tb:.1f} tok/s"
        if args.ar_baseline_ms is not None:
            _line += f"   speedup(vs AR {args.ar_baseline_ms:.3f} ms/tok) = {args.ar_baseline_ms / (tb * 1000):.2f}"
        print(_line)
    else:
        t1 = np.mean([ans["choices"][0]["decode_times"][0] / max(1, ans["choices"][0]["new_tokens"][0]) for ans in answers if ans["choices"][0]["new_tokens"]])
        out["ar_ms_per_tok"] = t1 * 1000
        print(f"Decoding speedup: {t1 / tb:.2f}")
        # RAW per-token times: the speedup RATIO is normalized to THIS run's own b=1 baseline (t1), which
        # varies across runs -> NOT comparable across configs. Compare bK_spec (or spec_throughput) directly.
        print(f"[timing] b1_AR = {t1 * 1000:.3f} ms/tok   bK_spec = {tb * 1000:.3f} ms/tok   "
              f"spec_throughput = {1.0 / tb:.1f} tok/s   <- COMPARE THIS across runs, not the speedup ratio")

    # Both throughput definitions, side by side, so the pairing with tau is never ambiguous.
    print(f"[throughput] POOLED = {out['throughput_pooled']:.1f} tok/s (sum tokens / sum time; pairs with "
          f"tau/step)   per-sample = {1.0 / tb:.1f} tok/s (mean of per-sample rates; pairs with tau/sample)")

    acceptance_lengths = list(chain(*[ans["choices"][1]["acceptance_lengths"][0] for ans in answers if ans["choices"][1]["acceptance_lengths"]]))
    tau_per_step = np.mean(acceptance_lengths) if acceptance_lengths else 0
    tau_per_sample = np.mean([np.mean(ans["choices"][1]["acceptance_lengths"][0]) for ans in answers if ans["choices"][1]["acceptance_lengths"]])
    out["tau_step"] = float(tau_per_step)
    out["tau_sample"] = float(tau_per_sample)
    print(f"Average Acceptance length (per step):  {tau_per_step:.2f}")
    print(f"Average Acceptance length (per sample): {tau_per_sample:.2f}")
    if not verbose:
        return out

    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    _n = len(acceptance_lengths)
    if _n:
        _apos = [sum(1 for a in acceptance_lengths if a >= k + 2) / _n for k in range(block_size - 1)]
        print(f"Per-position accept rate (accept_rate@k): {[f'{r:.4f}' for r in _apos]}")
    return out


def agg(vals, fmt="%.1f"):
    a = sorted(vals)
    n = len(a)
    m = sum(a) / n
    sd = (sum((x - m) ** 2 for x in a) / (n - 1)) ** 0.5 if n > 1 else 0.0
    med = a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2
    return {"mean": fmt % m, "std": fmt % sd, "median": fmt % med,
            "min": fmt % a[0], "max": fmt % a[-1], "_mean": m, "_median": med}


def print_latency_profile(xpress_refiner, markov_refiner):
    import model.dflash as _df

    def _stats(us):
        if not us:
            return None
        s = sorted(us)
        n = len(s)
        return dict(n=n, median=s[n // 2], mean=sum(s) / n,
                    p90=s[min(n - 1, int(0.90 * n))], min=s[0], max=s[-1])

    _which = ("markov" if markov_refiner is not None else
              "xpress" if xpress_refiner is not None else "none (drafter-only)")
    _rows = [("drafter forward", _stats(_df._DRAFTER_T)),
             ("base lm_head",    _stats(_df._BASE_T)),
             (f"refiner [{_which}]", _stats(_df._REFINER_T))]
    print("\n================= per-block LATENCY PROFILE (us) =================")
    print(f"{'stage':<26}{'n':>7}{'median':>10}{'mean':>10}{'p90':>10}{'min':>10}{'max':>10}")
    _tot = 0.0
    for _name, _s in _rows:
        if _s is None:
            print(f"{_name:<26}{'-':>7}{'(not recorded)':>50}")
            continue
        _tot += _s["median"]
        print(f"{_name:<26}{_s['n']:>7}{_s['median']:>10.1f}{_s['mean']:>10.1f}"
              f"{_s['p90']:>10.1f}{_s['min']:>10.1f}{_s['max']:>10.1f}")
    if _tot > 0:
        print("-" * 65)
        for _name, _s in _rows:
            if _s is not None:
                print(f"  {_name:<24} = {100.0 * _s['median'] / _tot:5.1f}% of the draft-side total")
        print(f"  draft-side total (median sum) = {_tot:.1f} us/block")
    print("NOTE: profiling adds a cuda.synchronize per stage per block -> THROUGHPUT above is perturbed.")
    print("=================================================================")
