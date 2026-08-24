import os
import torch
from transformers import AutoConfig
from model.dflash import DFlashDraftModel


def normalize_draft_config_for_benchmark(config):
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    config.dflash_config = dflash_config
    return config


def load_draft_model_for_benchmark(model_name_or_path: str, attn_impl: str):
    draft_config = AutoConfig.from_pretrained(model_name_or_path)
    draft_config = normalize_draft_config_for_benchmark(draft_config)
    return DFlashDraftModel.from_pretrained(
        model_name_or_path,
        config=draft_config,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )


def spec_gen(draft_model, target, tokenizer, args, input_ids, max_new_tokens, bs, runtime):
    return draft_model.spec_generate(
        target=target, input_ids=input_ids, max_new_tokens=max_new_tokens, block_size=bs,
        stop_token_ids=[tokenizer.eos_token_id], temperature=args.temperature,
        draft_temperature=args.draft_temperature,
        return_dict=True, **runtime)


def warmup_pipeline(dataset, tokenizer, target, draft_model, args, bs_list, runtime):
    _dry_mnt = min(512, args.max_new_tokens)
    _warm = [(dataset[0]["turns"][0], 64)]
    for _i in range(min(2, len(dataset))):
        _warm.append((dataset[_i]["turns"][0], _dry_mnt))
    with torch.inference_mode():
        for _bs in bs_list:
            for _content, _mnt in _warm:
                _w = tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": _content}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False),
                    return_tensors="pt").to(target.device)
                spec_gen(draft_model, target, tokenizer, args, _w, _mnt, _bs, runtime)


def report_compile_cache(t_warm):
    from torch._dynamo.utils import counters
    c = counters["inductor"]
    hit  = c.get("fxgraph_cache_hit", 0)
    miss = c.get("fxgraph_cache_miss", 0)
    byp  = c.get("fxgraph_cache_bypass", 0)
    werr = c.get("fxgraph_cache_write_error", 0)
    tot = hit + miss + byp
    rate = f"{100.0 * hit / tot:.0f}%" if tot else "n/a"
    print(f"[compile-cache] fxgraph hit={hit} miss={miss} bypass={byp} -> reuse {rate}"
          f"   (warmup wall time {t_warm:.1f}s, dir={os.environ.get('TORCHINDUCTOR_CACHE_DIR')})")
    if werr:
        print(f"[compile-cache] WARNING {werr} write errors -> cache is NOT being populated "
              f"(shared-FS permissions/quota?); every run will recompile.")
    if byp:
        print(f"[compile-cache] WARNING {byp} graphs BYPASSED the cache -> they recompile every process, "
              f"permanently. Run with TORCH_LOGS=+torch._inductor.codecache to see the bypass reason.")
    if tot and miss > hit:
        print(f"[compile-cache] NOTE more misses than hits -- expected on the FIRST run on this "
              f"node/torch/GPU combo (the cache key includes all three); re-run to confirm reuse.")


def setup_compile_all(target, draft_model, args, is_main):
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", args.inductor_cache_dir)
    torch._inductor.config.fx_graph_cache = True
    for _lim in ("cache_size_limit", "recompile_limit", "accumulated_recompile_limit"):
        if hasattr(torch._dynamo.config, _lim):
            setattr(torch._dynamo.config, _lim, 256 if "accumulated" in _lim else 64)
    if is_main:
        print(f"[compile-all] torch.compile target + drafter + refiner (dynamic=True). "
              f"inductor cache = {os.environ['TORCHINDUCTOR_CACHE_DIR']} "
              f"(first run ~10min; later runs reuse -> fast).")
    target = torch.compile(target, dynamic=True)
    draft_model.forward = torch.compile(draft_model.forward, dynamic=True)
    return target
