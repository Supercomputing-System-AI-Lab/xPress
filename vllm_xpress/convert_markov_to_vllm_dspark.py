#!/usr/bin/env python3
"""Convert OUR cotrained Markov checkpoint (refiner_cotrain_*.pt from the
hybrid-markov runs) into a vLLM-loadable DSpark draft-model directory, so the
in-vLLM three-way comparison (dflash / dspark / xpress) uses the SAME cotrained
drafter + the SAME Markov head as the paper's HF numbers.

    python convert_markov_to_vllm_dspark.py \
        --ckpt /work/.../hybrid-s1-b16warm-markov-final/refiner_cotrain_403k.pt \
        --base-config <z-lab dflash snapshot dir> \
        --out /work/.../Qwen3-8B-DSparkOurs-b16
"""
import argparse, json, pathlib
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base-config", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

tensors = {k: v.contiguous() for k, v in ck["draft_state_dict"].items()}
head = ck["refiner_state_dict"]
tensors["markov_head.markov_w1.weight"] = head["w1.weight"].contiguous()
tensors["markov_head.markov_w2.weight"] = head["w2.weight"].contiguous()
save_file(tensors, str(out / "model.safetensors"))

cfg_path = pathlib.Path(args.base_config)
if cfg_path.is_dir():
    cfg_path = cfg_path / "config.json"
cfg = json.loads(cfg_path.read_text())
cfg.update({
    "architectures": ["Qwen3DSparkModel"],
    "markov_rank": int(head["w1.weight"].shape[1]),
    "sample_from_anchor": False,          # fill-in layout (z-lab convention, ours)
})
(out / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"wrote {out}  ({len(tensors)} tensors, markov_rank={head['w1.weight'].shape[1]})")
