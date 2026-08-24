#!/bin/bash
set -euo pipefail

TARGET=Qwen/Qwen3-8B
DRAFT=z-lab/Qwen3-8B-DFlash-b16
XPRESS_CKPT="${XPRESS_CKPT:-UIUC-SSAIL/Qwen3-8B-XPress-b16}"
MK_CKPT="${MK_CKPT:-UIUC-SSAIL/Qwen3-8B-Markov-b16}"
DATASET="${1:-gsm8k}"
TEMP="${2:-0}"                        # 0 = greedy (published protocol), 1 = lossless sampling
if [ "$TEMP" = "0" ]; then _KDEF=4,5,6,7; _RDEF=6; else _KDEF=6,7,8; _RDEF=5; fi
K_LIST="${K_LIST:-$_KDEF}"            # env-overridable: K_LIST=6 ./bench_fair_fast.sh mt-bench 1
DRAFT_T="${DRAFT_T:-$TEMP}"           # advanced: DRAFT_T=0 also switches the DRAFT to argmax (dFlash conv.)
BLOCK=16
MAX_NEW_TOKENS=2048
MAX_SAMPLES=128
ROUNDS="${ROUNDS:-$_RDEF}"   # = WARMUP + variants, so the rotation is exactly balanced
WARMUP_ROUNDS=1
RUN_AR=0
OUTDIR="${OUTDIR:-./results}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE:-./inductor_cache}
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")"
mkdir -p "$OUTDIR"
NODE=$(hostname)
SUFFIX="${DRAFT_T:+_draftT$DRAFT_T}"
LOG="$OUTDIR/fairfast_T${TEMP}${SUFFIX}_${DATASET}_K${K_LIST//,/-}_${NODE}.log"

echo "==================================================================="
echo " FAIR interleaved (single process)   node=$NODE   dataset=$DATASET   K=$K_LIST"
echo " rounds=$ROUNDS (first $WARMUP_ROUNDS discarded)   order ROTATES each round"
python -c "from transformers.utils import is_flash_attn_2_available as f; \
print(' attention backend: ' + ('flash_attention_2' if f() else 'sdpa  <-- absolute tok/s NOT comparable to FA2 nodes'))"
echo " log: $LOG"
echo "==================================================================="

DMON="$OUTDIR/dmon_${DATASET}_${NODE}_$(date +%m%d_%H%M).log"
nvidia-smi dmon -i "${CUDA_VISIBLE_DEVICES}" -s puct -o T > "$DMON" 2>/dev/null &
DMON_PID=$!
trap 'kill $DMON_PID 2>/dev/null' EXIT
echo " dmon: $DMON"

if [ "$RUN_AR" = "1" ]; then
  echo ""
  echo ">>> b=1 AR baseline for $DATASET (same node, same session, same prompts)"
  python benchmark_compile_all.py \
    --model-name-or-path "$TARGET" \
    --draft-name-or-path "$DRAFT" \
    --block-size "$BLOCK" \
    --dataset "$DATASET" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-samples "$MAX_SAMPLES" \
    --compile-all \
    --inductor-cache-dir "$TORCHINDUCTOR_CACHE_DIR" \
    --temperature "$TEMP" \
    --ar-only \
    2>&1 | tee "$OUTDIR/ar_${DATASET}_${NODE}.log" | grep -E "AR-only|throughput"
  echo ">>> use the POOLED AR number above as the speedup denominator for $DATASET"
fi

python benchmark_compile_all.py \
  --model-name-or-path "$TARGET" \
  --draft-name-or-path "$DRAFT" \
  --xpress-refiner-path "$XPRESS_CKPT" \
  --markov-refiner-path "$MK_CKPT" \
  --fair-interleave "$ROUNDS" \
  --fair-warmup-rounds "$WARMUP_ROUNDS" \
  --fair-k-list "$K_LIST" \
  --block-size "$BLOCK" \
  --dataset "$DATASET" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-samples "$MAX_SAMPLES" \
  --use-graph --compile-all \
  --inductor-cache-dir "$TORCHINDUCTOR_CACHE_DIR" \
  --temperature "$TEMP" \
  ${DRAFT_T:+--draft-temperature "$DRAFT_T"} \
  --skip-ar \
  --answer-file "$OUTDIR/${DATASET}_fairfast_T${TEMP}${SUFFIX}.jsonl" \
  2>&1 | tee "$LOG"

echo ""
echo "saved -> $LOG"
