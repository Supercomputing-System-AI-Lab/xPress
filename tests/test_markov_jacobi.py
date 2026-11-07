"""Correctness of the Markov+Jacobi control (refiners/markov.py).

Run from the repository root:  python tests/test_markov_jacobi.py

The Markov head conditions position k on the token id at k-1 and nothing else, so a
block is a chain and sequential decoding solves it exactly. Jacobi solves the same
fixed point in parallel, which gives four weight-independent properties -- they hold
for any weights, so random weights on a small vocabulary exercise them exactly as the
released checkpoint does, on CPU, in seconds:

  1. K = block_size - 1 reproduces the sequential decode TOKEN FOR TOKEN.
  2. The emitted token is the argmax of the logits reported as q (the pairing the
     rejection-sampling guarantee rests on).
  3. After K passes the first K positions already hold their final values.
  4. K = 0 is the bare drafter seed.

(1) is what makes the control a clean isolation of the head: the ceiling of
Markov+Jacobi is the sequential Markov number, so the schedule cannot lift the
baseline past its own exact solution. Properties 1, 3 and 4 are checked with prefix
locking both off and on.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refiners.deepspec.markov_head import VanillaMarkov  # noqa: E402
from refiners.markov import markov_par_k_refine  # noqa: E402

VOCAB, RANK, BLOCK, BATCH, TRIALS = 257, 16, 16, 4, 10
DRAFT_SLOTS = BLOCK - 1


def _head():
    torch.manual_seed(0)
    return VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).double().eval()


def _case(trial):
    g = torch.Generator().manual_seed(1000 + trial)
    base = torch.randn(BATCH, DRAFT_SLOTS, VOCAB, generator=g, dtype=torch.float64) * 2.0
    anchor = torch.randint(0, VOCAB, (BATCH,), generator=g)
    return base, anchor


def main():
    head = _head()
    argmax = lambda logits: torch.argmax(logits, dim=-1)  # noqa: E731
    failures = []

    for trial in range(TRIALS):
        base, anchor = _case(trial)
        seq, _ = head.sample_block_tokens(
            base, first_prev_token_ids=anchor, hidden_states=None, temperature=0.0
        )
        for lock in (False, True):
            draft, q_logits = markov_par_k_refine(
                head, base, anchor, DRAFT_SLOTS, argmax, lock=lock
            )
            if not torch.equal(draft, seq):
                row = int((draft != seq).any(dim=1).nonzero()[0])
                failures.append(
                    f"trial {trial} lock={lock}: K={DRAFT_SLOTS} != sequential on row {row}\n"
                    f"    sequential {seq[row].tolist()}\n"
                    f"    jacobi     {draft[row].tolist()}"
                )
            picked = q_logits.gather(-1, draft.unsqueeze(-1)).squeeze(-1)
            if not torch.allclose(picked, q_logits.max(dim=-1).values):
                failures.append(f"trial {trial} lock={lock}: emitted token is not argmax(q)")

    print(f"[1] K=block-1 == sequential, and token == argmax(q), {TRIALS} trials x 2 lock modes: "
          f"{'PASS' if not failures else f'{len(failures)} FAILURES'}")
    for f in failures:
        print("   ", f)

    base, anchor = _case(TRIALS)
    seq, _ = head.sample_block_tokens(
        base, first_prev_token_ids=anchor, hidden_states=None, temperature=0.0
    )
    unsettled = []
    for lock in (False, True):
        for passes in range(1, DRAFT_SLOTS + 1):
            draft, _ = markov_par_k_refine(head, base, anchor, passes, argmax, lock=lock)
            if not bool((draft[:, :passes] == seq[:, :passes]).all()):
                unsettled.append(f"lock={lock} K={passes}")
    print(f"[2] monotone settling (first K positions final after K passes): "
          f"{'PASS' if not unsettled else 'FAILED at ' + ', '.join(unsettled)}")
    failures += unsettled

    seeds = [
        markov_par_k_refine(head, base, anchor, 0, argmax, lock=lock)[0] for lock in (False, True)
    ]
    seed_ok = all(torch.equal(s, argmax(base)) for s in seeds)
    print(f"[3] K=0 == the drafter's own argmax: {'PASS' if seed_ok else 'FAIL'}")
    if not seed_ok:
        failures.append("K=0 is not the drafter seed")

    for passes in range(1, DRAFT_SLOTS + 1):
        draft, _ = markov_par_k_refine(head, base, anchor, passes, argmax, lock=False)
        if torch.equal(draft, seq):
            print(f"[i] random weights converge at K={passes} (block={BLOCK}, worst case "
                  f"{DRAFT_SLOTS}); the real checkpoint is slower -- that curve is the measurement")
            break

    if failures:
        raise SystemExit(f"{len(failures)} failure(s)")
    print("all checks passed")


if __name__ == "__main__":
    main()
