import torch
import torch.nn as nn


class ChannelWiseCausalMix(nn.Module):
    """Per-channel learned lower-triangular token mix: u[b,k,c] = sum_{j<=k} L[c,k,j] x[b,j,c]."""
    def __init__(self, channels: int, block_size: int):
        super().__init__()
        self.L = nn.Parameter(torch.zeros(channels, block_size, block_size))
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.folded = False

    def forward(self, x):
        Lm = self.L if self.folded else self.L * self.tril 
        return torch.bmm(Lm, x.permute(2, 1, 0)).permute(2, 1, 0)


class RDimMLP(nn.Module):
    """SwiGLU MLP in the r-dim bottleneck."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class XPressRefinerHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, block_size: int, markov_rank: int = 256,
                 mlp_ratio: int = 2, initializer_range: float = 0.02):
        super().__init__()
        r = markov_rank
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.markov_rank = r
        self.initializer_range = float(initializer_range)
        self.mixer_folded = False

        self.w1 = nn.Embedding(vocab_size, r) 
        self.w2 = nn.Linear(r, vocab_size, bias=False)

        self.down_h = nn.Linear(hidden_size, r, bias=False)
        self.down_g = nn.Linear(hidden_size, r, bias=False)
        self.in_proj = nn.Linear(3 * r, r, bias=False)
        self.mix = ChannelWiseCausalMix(r, block_size)
        self.mlp = RDimMLP(r, r * mlp_ratio)

        self.reset_parameters()

    def reset_parameters(self):
        std = self.initializer_range
        for w in (self.w1.weight, self.w2.weight, self.down_h.weight, self.down_g.weight,
                  self.in_proj.weight, self.mlp.gate_proj.weight, self.mlp.up_proj.weight,
                  self.mlp.down_proj.weight):
            nn.init.normal_(w, mean=0.0, std=std)

    def refine_hidden_cache(self, h, g):
        return torch.cat([self.down_h(h), self.down_g(g)], dim=-1)

    def _refine(self, markov_latent, h, g, hcache=None):
        if hcache is None:
            hcache = self.refine_hidden_cache(h, g)
        x = self.in_proj(torch.cat([hcache, markov_latent], dim=-1))
        m = self.mix(x)
        x = m if self.mixer_folded else x + m
        return x + self.mlp(x)

    @torch.no_grad()
    def fold_mixer_(self):
        """
        The sublayer is x_out = x + (L*tril) x = (I + L*tril) x, so both the causal mask and the
        identity can be baked into the parameter ONCE: L <- L*tril + I ,   x_out = bmm(L, x)
        """
        if self.mixer_folded:
            return self
        L = self.mix.L
        Lf = L * self.mix.tril
        Lf = Lf + torch.eye(Lf.shape[-1], dtype=Lf.dtype, device=Lf.device)
        L.copy_(Lf)
        self.mix.folded = True
        self.mixer_folded = True
        return self

    def compute_latent(self, h, g, prev_token_ids, hcache=None):
        markov_latent = self.w1(prev_token_ids.long())
        return self._refine(markov_latent, h, g, hcache=hcache)

    def bias(self, h, g, prev_token_ids, hcache=None):
        return self.w2(self.compute_latent(h, g, prev_token_ids, hcache=hcache))

    def forward(self, base_logits, h, g, prev_token_ids, hcache=None):
        return base_logits, base_logits + self.bias(h, g, prev_token_ids, hcache=hcache)
