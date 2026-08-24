import argparse, json, pathlib
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base-config", required=True,
                help="dir containing the z-lab DFlash config.json (HF snapshot)")
ap.add_argument("--out", required=True)
ap.add_argument("--num-passes", type=int, default=6)
args = ap.parse_args()

out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)

# ---- weights: drafter (co-trained) + head under the xpress_head. prefix ----
tensors = {}
for k, v in ck["draft_state_dict"].items():
    tensors[k] = v.contiguous()
head_sd = ck["refiner_state_dict"]
KEEP = {"w1.weight", "down_h.weight", "down_g.weight", "in_proj.weight",
        "mix.L", "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight", "w2.weight"}
for k, v in head_sd.items():
    if k in KEEP:
        tensors["xpress_head." + k] = v.contiguous()
missing = KEEP - {k for k in head_sd}
assert not missing, f"checkpoint missing head keys: {missing}"
save_file(tensors, str(out / "model.safetensors"))

# ---- config: base DFlash config + xpress fields ----
cfg_path = pathlib.Path(args.base_config)
if cfg_path.is_dir():
    cfg_path = cfg_path / "config.json"
cfg = json.loads(cfg_path.read_text())
r = head_sd["w1.weight"].shape[1]
mlp_hidden = head_sd["mlp.gate_proj.weight"].shape[0]
block = head_sd["mix.L"].shape[-1]
cfg.update({
    "architectures": ["Qwen3XPressModel"],
    "xpress_rank": int(r),
    "xpress_mlp_hidden": int(mlp_hidden),
    "xpress_block_size": int(block),
    "xpress_num_passes": int(args.num_passes),
    "sample_from_anchor": False,      # fill-in block layout (z-lab convention)
})
(out / "config.json").write_text(json.dumps(cfg, indent=2))
step = ck.get("step", "?")
print(f"wrote {out}  ({len(tensors)} tensors, head r={r} mlp={mlp_hidden} "
      f"block={block}, cotrain step={step})")
