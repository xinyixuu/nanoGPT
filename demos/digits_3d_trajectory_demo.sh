#!/usr/bin/env bash
# Train a width-3 model and export every saved embedding snapshot for Three.js.
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
MAX_ITERS="${MAX_ITERS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
OUT_DIR="${OUT_DIR:-out/digits_3d}"
DATA_DIR="data/digits_3d"
VIEW_DIR="report/threejs/digits-3d"
WTE_FIXED_NORM="${WTE_FIXED_NORM:-true}"
WTE_FIXED_NORM_VALUE="${WTE_FIXED_NORM_VALUE:-}"

case "${WTE_FIXED_NORM}" in
  true|1|yes) WTE_NORM_ARGS=(--wte_fixed_norm) ;;
  false|0|no) WTE_NORM_ARGS=(--no-wte_fixed_norm) ;;
  *) echo "WTE_FIXED_NORM must be true or false" >&2; exit 2 ;;
esac
if [ -n "${WTE_FIXED_NORM_VALUE}" ]; then
  WTE_NORM_ARGS+=(--wte_fixed_norm_value "${WTE_FIXED_NORM_VALUE}")
fi

python3 "${DATA_DIR}/prepare.py"

python3 train.py \
  --dataset digits_3d \
  --out_dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --block_size 10 \
  --batch_size 64 \
  --n_layer 1 \
  --n_head 1 \
  --n_embd 3 \
  "${WTE_NORM_ARGS[@]}" \
  --dropout 0.0 \
  --max_iters "${MAX_ITERS}" \
  --eval_interval "${SAVE_INTERVAL}" \
  --eval_iters 20 \
  --save_major_ckpt_interval "${SAVE_INTERVAL}" \
  --always_save_checkpoint \
  --learning_rate 3e-3 \
  --min_lr 3e-4 \
  --warmup_iters 20 \
  --decay_lr \
  --no-compile

python3 analysis/export_3d_token_trajectories.py \
  --checkpoint-dir "${OUT_DIR}" \
  --meta "${DATA_DIR}/meta.pkl" \
  --output "${VIEW_DIR}/token_trajectories.json"

cat <<EOF
Done. Serve the repository (fetch does not work from file://), then open:
  python3 -m http.server 8000
  http://localhost:8000/${VIEW_DIR}/index.html
Digits are trained (warm colors); a-d are never sampled (blue controls).
EOF
