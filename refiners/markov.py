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
