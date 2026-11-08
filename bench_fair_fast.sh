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
# ALSO run the Markov head under the same Jacobi schedule, one variant per K, in
# the same rotation -- the schedule-only control (identical seed, identical prefix
# locking, only the head differs). Empty = off, i.e. the published protocol, where
# Markov appears once and is decoded sequentially.
#   MK_K_LIST=1,2,4,7,10,15 ./bench_fair_fast.sh gsm8k 0
# Include K=block-1 (15): the Markov chain conditions position k on the token id at
# k-1 alone, so that K reproduces the sequential decode token for token and its tau
# must equal the "Markov" row. That equality is the run's correctness check.
MK_K_LIST="${MK_K_LIST:-}"
DRAFT_T="${DRAFT_T:-$TEMP}"           # advanced: DRAFT_T=0 also switches the DRAFT to argmax (dFlash conv.)
BLOCK=16
MAX_NEW_TOKENS=2048
MAX_SAMPLES=128              # AIME25 contains only **30 problems**, so please use `--max-samples 30`
# = WARMUP + variants, so the rotation is exactly balanced: each variant leads
# exactly once over the measured rounds. Variants = |K_LIST| XPress + |MK_K_LIST|
# Markov+Jacobi + 1 sequential Markov, so adding controls widens the default.
_NK=$(awk -F, '{print NF}' <<< "$K_LIST")
_NMK=0; [ -n "$MK_K_LIST" ] && _NMK=$(awk -F, '{print NF}' <<< "$MK_K_LIST")
if [ -n "$MK_K_LIST" ]; then _RDEF=$((1 + _NK + _NMK + 1)); fi
ROUNDS="${ROUNDS:-$_RDEF}"
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
[ -n "$MK_K_LIST" ] && echo " Markov+Jacobi control K=$MK_K_LIST  (K=$((BLOCK - 1)) must match the sequential Markov row)"
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
  ${MK_K_LIST:+--fair-markov-k-list "$MK_K_LIST"} \
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
