import torch

def _load_vanilla_markov():
    """DeepSpec's VanillaMarkov, vendored under refiners/deepspec/."""
    from refiners.deepspec.markov_head import VanillaMarkov
    return VanillaMarkov

def load_markov_head(checkpoint_path, vocab_size, device, dtype=torch.bfloat16):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", None)
    markov_rank = int(getattr(args, "markov_rank", 256)) if args is not None else 256

    VanillaMarkov = _load_vanilla_markov()
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)

    sd = ckpt["refiner_state_dict"]
    remap = {}
    if "w1.weight" in sd:
        remap["markov_w1.weight"] = sd["w1.weight"]
    if "w2.weight" in sd:
        remap["markov_w2.weight"] = sd["w2.weight"]
    missing, unexpected = head.load_state_dict(remap, strict=False)
    core_missing = [k for k in missing if k.startswith(("markov_w1", "markov_w2"))]
    if core_missing:
        print(f"[markov] MISSING core markov weights (left at init -> garbage!): {core_missing}", flush=True)
    head = head.to(device).to(dtype).eval()
    print(f"[markov] loaded FAITHFUL DeepSpec VanillaMarkov: rank={markov_rank} "
          f"step={ckpt.get('global_step')} from {checkpoint_path} (w1/w2 -> markov_w1/markov_w2)", flush=True)
    return head


def _bcast(t, bs):
    return t if t.shape[0] == bs else t.expand(bs, *t.shape[1:])


class MarkovSeqGraphRunner:
    """CUDA-graph the SEQUENTIAL DeepSpec markov decode (fixed block, GREEDY)."""
    def __init__(self, markov_head, block_size, vocab_size, device, dtype=torch.bfloat16, bs=1,
                 temperature=0.0):
        self.head = markov_head
        self.block_size = int(block_size)
        self.bs = int(bs)
        self.temperature = float(temperature)
        self.static_base_logits = torch.zeros(self.bs, block_size - 1, vocab_size, dtype=dtype, device=device)
        self.static_anchor_id = torch.zeros(self.bs, dtype=torch.long, device=device)
        self.static_out = torch.zeros(self.bs, block_size - 1, dtype=torch.long, device=device)
        self.static_q = (torch.zeros(self.bs, block_size - 1, vocab_size, dtype=dtype, device=device)
                         if self.temperature > 0 else None)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):
                self._forward_impl()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._forward_impl()

    def _forward_impl(self):
        sampled, logits = self.head.sample_block_tokens(
            self.static_base_logits,
            first_prev_token_ids=self.static_anchor_id,
            hidden_states=None,
            temperature=self.temperature,
        )
        if self.static_q is not None:
            self.static_q.copy_(logits)
        self.static_out.copy_(sampled)

    @torch.inference_mode()
    def __call__(self, base_logits, anchor_id):
        self.static_base_logits.copy_(_bcast(base_logits, self.bs))
        self.static_anchor_id.copy_(_bcast(anchor_id.reshape(-1), self.bs))
        self.graph.replay()
        return self.static_out.clone()


# ----------------------------------------------------------------------------------
# JACOBI rollout of the Markov head -- the schedule-only control for XPress.
#
# The Markov head conditions position k on the token id at k-1 and nothing else, so a
# block is a chain y_1 <- anchor, y_2 <- y_1, ... Sequential decoding solves that chain
# exactly in B-1 ordered steps; Jacobi solves the same fixed point by updating every
# position in parallel from the current iterate. Under a deterministic sample_fn the
# prefix settles one position per pass (position k's predecessor is final after pass
# k-1), so K = B-1 reproduces the sequential decode TOKEN FOR TOKEN and every K < B-1
# is a truncation of it. The ceiling of this control is therefore the published
# sequential Markov number, which is what makes it a clean isolation of the head: the
# schedule cannot lift the baseline past its own exact solution.
#
# Deliberately mirrors refiners/xpress.py (same seed, same prefix locking, same "q =
# the last pass's corrected logits") so that the two curves differ only in the head.
# ----------------------------------------------------------------------------------


def _markov_prev(anchor_id, blk):
    """prev[:, k] = the token position k conditions on: anchor for k=0, else blk[k-1].

    Matches VanillaMarkov.sample_block_tokens, which starts from first_prev_token_ids
    and then feeds its own last sample.
    """
    return torch.cat([anchor_id.reshape(-1, 1).long(), blk[:, :-1]], dim=1)


@torch.inference_mode()
def markov_par_k_refine(head, base_logits, anchor_id, refine_passes, sample_fn, lock=False):
    """K parallel Jacobi passes over the Markov head. Returns (draft ids, q logits).

    base_logits: [bs, block-1, V] the drafter's own logits for the draft slots
    anchor_id:   [bs] the bonus token the block is anchored on
    sample_fn:   logits [bs, L, V] -> ids [bs, L]; must be DETERMINISTIC given the
                 logits (argmax at T=0, frozen-Gumbel argmax at T>0), otherwise the
                 iterate never settles and the proposed block is not a coherent chain.
    """
    blk = sample_fn(base_logits)              # seed = the drafter's own draw, as in XPress
    refined = base_logits
    locked = None                             # per-row prefix mask (xpress.py's eager form is bs=1 only)
    for _ in range(refine_passes):
        refined = head.apply_block_logits(
            base_logits, token_ids=_markov_prev(anchor_id, blk), hidden_states=None)
        new_tok = sample_fn(refined)
        if lock:
            if locked is not None:
                new_tok = torch.where(locked, blk, new_tok)   # locked prefix never rewritten
            locked = (new_tok == blk).to(torch.int64).cumprod(dim=1).bool()
        blk = new_tok
        if lock and bool(locked.all()):
            break                                             # whole block converged
    return blk, refined


class MarkovJacobiGraphRunner:
    """CUDA-graph the JACOBI markov decode (fixed block, fixed K).

    Drop-in for MarkovSeqGraphRunner: identical __call__(base_logits, anchor_id) and
    static_q, so the decode loop in model/dflash.py needs no change -- only which
    runner object it is handed. Locking is the tensor form (no .item(), no early
    break) so the graph has static control flow, exactly as XPressRefineGraphRunner.
    """

    def __init__(self, markov_head, block_size, vocab_size, device, dtype=torch.bfloat16,
                 bs=1, temperature=0.0, refine_passes=1, lock=False):
        self.head = markov_head
        self.block_size = int(block_size)
        self.refine_passes = int(refine_passes)
        self.bs = int(bs)
        self.temperature = float(temperature)
        self.lock = bool(lock)
        L = self.block_size - 1
        self.static_base_logits = torch.zeros(self.bs, L, vocab_size, dtype=dtype, device=device)
        self.static_anchor_id = torch.zeros(self.bs, dtype=torch.long, device=device)
        self.static_out = torch.zeros(self.bs, L, dtype=torch.long, device=device)
        self.static_q = (torch.zeros(self.bs, L, vocab_size, dtype=dtype, device=device)
                         if self.temperature > 0 else None)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(s):
            for _ in range(3):
                self._forward_impl()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self._forward_impl()

    def _forward_impl(self):
        base = self.static_base_logits
        L = self.block_size - 1
        if self.temperature > 0:
            # FROZEN Gumbel noise: drawn ONCE per replay (= once per block), shared by the
            # seed and all K passes. Two-step Exp->Gumbel: `-torch.log(x).clamp_min(..)`
            # clamps BEFORE the unary minus (method binds tighter) and yields NaN noise.
            u = torch.rand(self.bs, L, base.shape[-1], device=base.device, dtype=torch.float32)
            e = -torch.log(u.clamp_min(1e-20))
            noise = -torch.log(e.clamp_min(1e-20))
            T = self.temperature
            samp = lambda lg: torch.argmax(lg.float() / T + noise, dim=-1)
        else:
            samp = lambda lg: torch.argmax(lg, dim=-1)

        blk = samp(base)
        locked = (torch.zeros(self.bs, L, dtype=torch.bool, device=base.device)
                  if self.lock else None)
        refined = base
        for _ in range(self.refine_passes):
            prev = torch.cat([self.static_anchor_id.unsqueeze(1), blk[:, :-1]], dim=1)
            refined = self.head.apply_block_logits(base, token_ids=prev, hidden_states=None)
            new_tok = samp(refined)
            if locked is not None:
                new_tok = torch.where(locked, blk, new_tok)
                locked = (new_tok == blk).to(torch.int64).cumprod(dim=1).bool()
            blk = new_tok
        if self.static_q is not None:
            self.static_q.copy_(refined)                   # q = last pass's corrected logits
        self.static_out.copy_(blk)

    @torch.inference_mode()
    def __call__(self, base_logits, anchor_id):
        self.static_base_logits.copy_(_bcast(base_logits, self.bs))
        self.static_anchor_id.copy_(_bcast(anchor_id.reshape(-1), self.bs))
        self.graph.replay()
        return self.static_out.clone()
