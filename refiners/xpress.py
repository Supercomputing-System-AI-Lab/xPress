import torch
from refiners.xpress_head import XPressRefinerHead


def _head_kwargs_from_args(a):
    g = lambda k, d: getattr(a, k, d)
    expected = dict(markov_only=False, no_hidden=False, no_perpos=False, no_global=False,
                    no_residual=True, no_mixer=False, no_mlp=False, input_mode="concat",
                    no_sublayer_norm=True, no_mix_out=True, lowrank_lmhead_shared_rank=0,
                    lowrank_base_rank=0, input_relu=False)
    mismatched = {k: g(k, v) for k, v in expected.items() if g(k, v) != v}
    if mismatched:
        raise RuntimeError(f"checkpoint was trained with a head variant not included in this "
                           f"release: {mismatched} (expected {expected})")
    return dict(
        markov_rank=int(g("markov_rank", 256)),
        mlp_ratio=int(g("mlp_ratio", 2)),
    )


def load_xpress_refiner(checkpoint_path, vocab_size, hidden_size, block_size, device, dtype=torch.bfloat16,
                        fold_mixer=True):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", None)
    if args is None:
        raise RuntimeError("checkpoint has no 'args' -> cannot rebuild the XPressRefinerHead config.")
    kw = _head_kwargs_from_args(args)
    head = XPressRefinerHead(vocab_size, hidden_size, block_size, **kw)

    sd = ckpt["refiner_state_dict"]
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if missing:
        print(f"[xpress refiner] MISSING keys (left at init!): {sorted(missing)}", flush=True)
    if unexpected:
        print(f"[xpress refiner] UNEXPECTED keys (ignored): {sorted(unexpected)}", flush=True)
    head = head.to(device).to(dtype).eval()
    if fold_mixer:
        head.fold_mixer_()
        print("[xpress refiner] mixer FOLDED for inference (L <- L*tril + I; residual add + mask dropped)",
              flush=True)
    print(f"[xpress refiner] loaded: head_rank={kw['markov_rank']} mlp_ratio={kw['mlp_ratio']} "
          f"step={ckpt.get('global_step')} from {checkpoint_path}", flush=True)
    return head


@torch.inference_mode()
def xpress_par_k_refine(head, h_full, base_logits, anchor_id, tok_am1, refine_passes, sample_fn,
                        lock=False):
    block = h_full.shape[1]
    h = h_full
    g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)
    base_full = torch.cat([base_logits[:, :1, :], base_logits], dim=1)

    hcache = head.refine_hidden_cache(h, g)
    blk = torch.empty(1, block, dtype=torch.long, device=h.device)
    blk[:, 0] = anchor_id.reshape(-1)
    blk[:, 1:] = sample_fn(base_full[:, 1:, :])

    locked = 0 
    for _ in range(refine_passes):
        prev = blk.roll(shifts=1, dims=1)
        prev[:, 0] = tok_am1.reshape(-1) 
        _, refined = head(base_full, h, g, prev, hcache=hcache)
        new_tok = sample_fn(refined[:, 1:, :])
        if lock:
            if locked > 0:
                new_tok[:, :locked] = blk[:, 1 : 1 + locked]
            same = (new_tok == blk[:, 1:]).to(torch.int64)
            locked += int(same[0, locked:].cumprod(0).sum().item())
        blk[:, 1:] = new_tok
        if lock and locked >= blk.shape[1] - 1:
            break 
    return blk[:, 1:]


class DrafterSeedGraphRunner:

    def __init__(self, block_size, vocab_size, device, dtype=torch.bfloat16, temperature=0.0, bs=1):
        self.temperature = float(temperature)
        self.bs = int(bs)
        self.static_base = torch.zeros(self.bs, block_size - 1, vocab_size, dtype=dtype, device=device)
        self.static_out = torch.zeros(self.bs, block_size - 1, dtype=torch.long, device=device)
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
        if self.temperature > 0:
            u = torch.rand(self.static_base.shape, device=self.static_base.device, dtype=torch.float32)
            e = -torch.log(u.clamp_min(1e-20))
            noise = -torch.log(e.clamp_min(1e-20))
            self.static_out.copy_(
                torch.argmax(self.static_base.float() / self.temperature + noise, dim=-1))
        else:
            self.static_out.copy_(torch.argmax(self.static_base, dim=-1))

    @torch.inference_mode()
    def __call__(self, base_logits):
        self.static_base.copy_(base_logits if base_logits.shape[0] == self.bs
                               else base_logits.expand(self.bs, -1, -1))
        self.graph.replay()
        return self.static_out.clone()


class XPressRefineGraphRunner:
    def __init__(self, head, block_size, refine_passes, hidden_dim, vocab_size,
                 device, dtype=torch.bfloat16, bs=1, temperature=0.0, lock=False):
        self.head = head
        self._mod = head._orig_mod if hasattr(head, "_orig_mod") else head
        self.block_size = int(block_size)
        self.refine_passes = int(refine_passes)
        self.bs = int(bs)
        self.temperature = float(temperature)
        self.lock = bool(lock)
        self.static_q = (torch.zeros(self.bs, block_size - 1, vocab_size, dtype=dtype, device=device)
                         if self.temperature > 0 else None)
        self.static_h_full = torch.zeros(self.bs, block_size, hidden_dim, dtype=dtype, device=device)
        self.static_anchor_id = torch.zeros(self.bs, dtype=torch.long, device=device)
        self.static_tok_am1 = torch.zeros(self.bs, dtype=torch.long, device=device)
        self.static_blk = torch.zeros(self.bs, block_size, dtype=torch.long, device=device)
        self.static_out = torch.zeros(self.bs, block_size - 1, dtype=torch.long, device=device)
        self.static_base_full = torch.zeros(self.bs, block_size, vocab_size, dtype=dtype, device=device)

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
        h = self.static_h_full
        block = self.block_size
        g = h.mean(dim=1, keepdim=True).expand(-1, block, -1)
        base_full = self.static_base_full
        hcache = self._mod.refine_hidden_cache(h, g)
        blk = self.static_blk
        blk[:, 0] = self.static_anchor_id
        if self.temperature > 0:
            # FROZEN Gumbel noise: drawn ONCE per replay (= once per block), shared by seed + all K
            # passes. Two-step Exp->Gumbel (the -log(x).clamp precedence trap bites here too).
            u = torch.rand(self.bs, self.block_size - 1, base_full.shape[-1],
                           device=blk.device, dtype=torch.float32)
            e = -torch.log(u.clamp_min(1e-20))
            noise = -torch.log(e.clamp_min(1e-20))
            T = self.temperature
            samp = lambda lg: torch.argmax(lg.float() / T + noise, dim=-1)
        else:
            samp = lambda lg: torch.argmax(lg, dim=-1)
        blk[:, 1:] = samp(base_full[:, 1:, :]) 
        locked = (torch.zeros(self.bs, self.block_size - 1, dtype=torch.bool, device=blk.device)
                  if self.lock else None)
        refined = None
        for _ in range(self.refine_passes):
            prev = blk.roll(shifts=1, dims=1)
            prev[:, 0] = self.static_tok_am1
            _, refined = self.head(base_full, h, g, prev, hcache=hcache)
            new_tok = samp(refined[:, 1:, :])
            if locked is not None:
                new_tok = torch.where(locked, blk[:, 1:], new_tok)
                locked = (new_tok == blk[:, 1:]).to(torch.int64).cumprod(dim=1).bool()
            blk[:, 1:] = new_tok
        if self.static_q is not None:
            self.static_q.copy_(refined[:, 1:, :])                    # q = last pass's refined logits
        self.static_out.copy_(blk[:, 1:])

    @torch.inference_mode()
    def __call__(self, h_full, base_logits, anchor_id, tok_am1):
        from refiners.markov import _bcast
        bs = self.bs
        self.static_h_full.copy_(_bcast(h_full, bs))
        self.static_base_full[:, 1:, :].copy_(_bcast(base_logits, bs))
        self.static_anchor_id.copy_(_bcast(anchor_id.reshape(-1), bs))
        self.static_tok_am1.copy_(_bcast(tok_am1.reshape(-1), bs))
        self.graph.replay()
        return self.static_out.clone()
