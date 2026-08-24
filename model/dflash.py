import os
import time
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


_DRAFTER_T = []
_BASE_T = []
_REFINER_T = []


def _time_block(tstart_evt, acc, tag):
    """helper: record elapsed us into `acc` and print median/mean every 200 (rank 0)."""
    tend = torch.cuda.Event(enable_timing=True); tend.record(); torch.cuda.synchronize()
    acc.append(tstart_evt.elapsed_time(tend) * 1000.0)
    if len(acc) % 200 == 0:
        _s = sorted(acc)
        print(f"[{tag}] median={_s[len(_s)//2]:.1f}us mean={sum(_s)/len(_s):.1f}us n={len(_s)}", flush=True)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits.view(-1, vocab_size).float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)

def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_residual(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """[B, V] x [B, V] -> [B] token from normalize(clamp(p - q, 0)) (DeepSpec sample_residual)."""
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return torch.multinomial(residual, num_samples=1).squeeze(-1)


def rejection_verify(target_logits, draft_logits, verify_ids, temperature, draft_temperature=None):
    if draft_temperature is None:
        draft_temperature = temperature
    target_probs = logits_to_probs(target_logits, float(temperature))       # [1, n+1, V]
    draft_probs = logits_to_probs(draft_logits, float(draft_temperature))   # [1, n,  V]
    proposed = verify_ids[:, 1:]                                            # [1, n]
    p_tok = torch.gather(target_probs[:, :-1, :], -1, proposed.unsqueeze(-1)).squeeze(-1)
    q_tok = torch.gather(draft_probs, -1, proposed.unsqueeze(-1)).squeeze(-1)
    # q(token)==0 (proposal impossible under q) must REJECT, never accept
    accept_prob = torch.where(q_tok > 0, torch.clamp(p_tok / q_tok.clamp_min(1e-8), max=1.0),
                              torch.zeros_like(p_tok))
    accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
    accepted = int(accept_mask.cumprod(dim=1).sum(dim=1)[0].item())
    if accepted < proposed.shape[1]:
        next_token = sample_residual(target_probs[:, accepted, :], draft_probs[:, accepted, :])
    else:
        next_token = torch.multinomial(target_probs[:, -1, :], num_samples=1).squeeze(-1)
    return accepted, next_token


class _QLogitsRecorder:
    def __init__(self, temperature):
        self.temperature = float(temperature)
        self.last_logits = None
        self.noise = None

    def reset(self):
        self.last_logits = None
        self.noise = None                      # fresh Gumbel noise per block

    def __call__(self, logits):
        self.last_logits = logits.detach()
        if self.temperature < 1e-5:
            return torch.argmax(logits, dim=-1)
        if self.noise is None or self.noise.shape != logits.shape:
            u = torch.rand(logits.shape, device=logits.device, dtype=torch.float32)
            e = -torch.log(u.clamp_min(1e-20))            # Exp(1) sample, >= 0
            self.noise = -torch.log(e.clamp_min(1e-20))   # Gumbel(0,1); two steps avoid the
            # unary-minus/method-precedence trap: -log(x).clamp() clamps BEFORE negation.
        return torch.argmax(logits.float() / self.temperature + self.noise, dim=-1)


def cuda_time(device: torch.device | str | int | None = None) -> float:
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            cuda_device = (
                torch.device(f"cuda:{device}")
                if isinstance(device, int)
                else torch.device(device)
            )
            if cuda_device.type == "cuda":
                torch.cuda.synchronize(cuda_device)
    return time.perf_counter()


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.pure_draft_prefix_len = self.config.dflash_config.get("pure_draft_prefix_len", 0)
        # Refiner heads (XPress / Markov) are attached externally; the drafter
        # itself carries no projector head.

        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        input_ids: torch.Tensor,
        target: nn.Module,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        draft_temperature: Optional[float] = None,
        stop_token_ids: Optional[list[int] | int] = None,
        block_size: Optional[int] = None,
        return_dict: bool = False,
        xpress_refine_passes: int = 3,
        xpress_refiner=None,          # XPress refiner head (par-K, logits-direct)
        markov_refiner=None,          # FAITHFUL DeepSpec VanillaMarkov (SEQUENTIAL sample_block_tokens)
        xpress_graph_runner=None,     # CUDA-graph for the XPress par-K rollout (greedy)
        markov_graph_runner=None,     # CUDA-graph for the markov sequential decode
        drafter_seed_graph_runner=None,  # CUDA-graph for the drafter-only K=0 seed sampling
        drafter_only: bool = False,
    ) -> torch.Tensor | SimpleNamespace:
        """Speculative generation with block drafting + refinement.

        This method currently supports a single sequence on one GPU, matching the
        draft checkpoints released with this repository.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "spec_generate currently supports input_ids with shape [1, seq_len]."
            )

        target_device = next(target.parameters()).device
        if target_device != self.device:
            raise ValueError(
                "The draft model and target model must be on the same device; "
                f"got draft={self.device}, target={target_device}."
            )

        input_ids = input_ids.to(self.device)
        block_size = int(block_size or self.block_size)
        mask_token_id = self.mask_token_id
        if mask_token_id is None:
            raise ValueError("The draft model config must define dflash_config.mask_token_id.")

        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        elif stop_token_ids is not None:
            stop_token_ids = list(stop_token_ids)

        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + int(max_new_tokens)
        extra_buffer = block_size

        output_ids = torch.full(
            (1, max_length + extra_buffer),
            mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        prefill_start = cuda_time(self.device)
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=block_size > 1,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )
        time_to_first_token = cuda_time(self.device) - prefill_start

        decode_start = cuda_time(self.device)
        start = num_input_tokens
        acceptance_lengths: list[int] = []
        draft_prefill = True
        prefix_len = int(self.pure_draft_prefix_len)

        draft_t = temperature if draft_temperature is None else float(draft_temperature)
        # q_recorder only when the draft actually SAMPLES (draft_t>0): rejection verify
        # against the recorded q. draft_t=0 -> greedy drafting; verification then uses
        # exact match against a sampled target posterior, which accepts each proposed
        # token with probability p(token) -- distributionally identical to rejection
        # against the one-hot q, and identical between the eager and graph paths.
        q_recorder = _QLogitsRecorder(draft_t) if (temperature >= 1e-5 and draft_t >= 1e-5) else None
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            k_draft = block_size - 1
            verify_ids = torch.full(
                (1, k_draft + 1),
                mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            verify_ids[:, 0] = output_ids[:, start]
            verify_position_ids = position_ids[:, start : start + k_draft + 1]

            use_xpress = False        # set below when block_size > 1; AR-only path never assigns it
            if block_size > 1:
                # drafter_only: plain DFlash drafter (no refiner) at K=0.
                use_xpress = (xpress_refiner is not None
                           or markov_refiner is not None
                           or drafter_only)

                noise_embedding = target.model.embed_tokens(block_output_ids)
                _td = os.environ.get("TIME_DRAFTER")
                if _td:
                    _ds = torch.cuda.Event(enable_timing=True); _de = torch.cuda.Event(enable_timing=True)
                    _ds.record()
                parallel_hiddens = self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[
                        :, past_key_values_draft.get_seq_length() : start + block_size
                    ],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                if _td:
                    _de.record(); torch.cuda.synchronize()
                    _DRAFTER_T.append(_ds.elapsed_time(_de) * 1000.0)   # us
                    if len(_DRAFTER_T) % 200 == 0:
                        _s = sorted(_DRAFTER_T)
                        print(f"[TIME_DRAFTER] drafter block forward: median={_s[len(_s)//2]:.1f}us "
                              f"mean={sum(_s)/len(_s):.1f}us n={len(_s)}", flush=True)
                past_key_values_draft.crop(start)

                if use_xpress:
                    if q_recorder is not None:
                        q_recorder.reset()
                    q_sample_fn = q_recorder if q_recorder is not None else (lambda lg: sample(lg, draft_t))
                    # XPress par-K refine: keep the FULL block hidden
                    # (anchor at [:, 0]) for the mean-pool g, seed from the drafter, then K parallel
                    # Jacobi passes. SAME verify/accept below -- only this refine step differs.
                    h_full = parallel_hiddens[:, -block_size:, :]
                    _tb = os.environ.get("TIME_BASE")
                    if _tb:
                        _bs_evt = torch.cuda.Event(enable_timing=True); _bs_evt.record()
                    base_logits = target.lm_head(h_full[:, 1:, :])
                    if _tb:
                        _time_block(_bs_evt, _BASE_T, "TIME_BASE base-lm_head")
                    _tr = os.environ.get("TIME_REFINER")
                    if _tr:
                        _rf_evt = torch.cuda.Event(enable_timing=True); _rf_evt.record()
                    if drafter_only:
                        # BASELINE: the draft IS the drafter's seed (K=0); takes precedence over
                        # any loaded refiner (with a refiner path, this is the co-trained drafter alone).
                        if drafter_seed_graph_runner is not None:
                            verify_ids[:, 1:] = drafter_seed_graph_runner(base_logits)
                        else:
                            verify_ids[:, 1:] = q_sample_fn(base_logits)
                        if q_recorder is not None:
                            q_recorder.last_logits = base_logits.detach()
                    elif markov_refiner is not None:
                        # DeepSpec Markov head. base_logits = lm_head(h_full[:, 1:]); anchor = prev of slot 0.
                        # The graph runner captures the same sequential decode.
                        if markov_graph_runner is not None:
                            verify_ids[:, 1:] = markov_graph_runner(base_logits, output_ids[:, start])
                            if q_recorder is not None and getattr(markov_graph_runner, "static_q", None) is not None:
                                q_recorder.last_logits = markov_graph_runner.static_q.clone()
                        else:
                            _mk_sampled, _mk_logits = markov_refiner.sample_block_tokens(
                                base_logits,
                                first_prev_token_ids=output_ids[:, start].reshape(-1),
                                hidden_states=None,
                                temperature=draft_t,
                            )
                            verify_ids[:, 1:] = _mk_sampled
                            if q_recorder is not None:
                                q_recorder.last_logits = _mk_logits.detach()
                    elif xpress_refiner is not None:
                        # XPress head: par-K Jacobi, head returns LOGITS directly
                        # (base + W2(latent)) -> no separate lm_head in the loop. Same verify/accept below.
                        if xpress_graph_runner is not None:
                            verify_ids[:, 1:] = xpress_graph_runner(
                                h_full, base_logits, output_ids[:, start], output_ids[:, start - 1])
                            if q_recorder is not None and getattr(xpress_graph_runner, "static_q", None) is not None:
                                q_recorder.last_logits = xpress_graph_runner.static_q.clone()
                        else:
                            from refiners.xpress import xpress_par_k_refine
                            verify_ids[:, 1:] = xpress_par_k_refine(
                                xpress_refiner,
                                h_full,
                                base_logits,
                                output_ids[:, start],        # anchor (verified token, block slot 0)
                                output_ids[:, start - 1],    # tok_am1 (token before the anchor)
                                xpress_refine_passes,
                                q_sample_fn,
                                lock=q_recorder is not None,
                            )
                    else:
                        raise RuntimeError(
                            "this release requires --xpress-refiner-path, "
                            "--markov-refiner-path, or --drafter-only"
                        )
                    if _tr:
                        _time_block(_rf_evt, _REFINER_T, "TIME_REFINER correction")
                else:
                    raise RuntimeError(
                        "this release requires --xpress-refiner-path, "
                        "--markov-refiner-path, or --drafter-only"
                    )
                if draft_prefill:
                    draft_prefill = False
                    decode_start = cuda_time(self.device)

            output = target(
                verify_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=block_size > 1,
            )

            _q = (q_recorder.last_logits
                  if (q_recorder is not None and use_xpress and block_size > 1)
                  else None)
            if _q is not None:
                # DeepSpec-faithful T>0 verify: rejection sampling against the refiner's q.
                acceptance_length, _next_tok = rejection_verify(
                    output.logits, _q, verify_ids, temperature, draft_temperature=draft_t)
                output_ids[:, start : start + acceptance_length + 1] = verify_ids[
                    :, : acceptance_length + 1
                ]
                output_ids[:, start + acceptance_length + 1] = _next_tok
            else:
                posterior = sample(output.logits, temperature)
                acceptance_length = (
                    (verify_ids[:, 1:] == posterior[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                output_ids[:, start : start + acceptance_length + 1] = verify_ids[
                    :, : acceptance_length + 1
                ]
                output_ids[:, start + acceptance_length + 1] = posterior[
                    :, acceptance_length
                ]

            acceptance_lengths.append(int(acceptance_length) + 1)
            start += int(acceptance_length) + 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(
                    output.hidden_states, self.target_layer_ids
                )[:, : acceptance_length + 1, :]

            if stop_token_ids is not None:
                stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
                if torch.isin(output_ids[:, num_input_tokens:start], stop_tensor).any():
                    break

        output_ids = output_ids[:, :max_length]
        _gen = output_ids[:, num_input_tokens:]
        output_ids = torch.cat([output_ids[:, :num_input_tokens],
                                _gen[:, _gen[0] != mask_token_id]], dim=1)
        if stop_token_ids is not None:
            stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_tensor
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0].item() + 1
                ]

        if not return_dict:
            return output_ids

        num_output_tokens = output_ids.shape[1] - num_input_tokens
        total_decode_time = cuda_time(self.device) - decode_start
        time_per_output_token = (
            total_decode_time / num_output_tokens if num_output_tokens > 0 else 0.0
        )
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=num_output_tokens,
            time_to_first_token=time_to_first_token,
            time_per_output_token=time_per_output_token,
            acceptance_lengths=acceptance_lengths,
        )
