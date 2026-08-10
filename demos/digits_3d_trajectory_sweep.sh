#!/usr/bin/env bash
# Run every embedding-radius mode across several trained/held-out vocab sizes.
set -euo pipefail

DIGIT_COUNTS="${DIGIT_COUNTS:-5 10 16}"
LETTER_COUNTS="${LETTER_COUNTS:-0 4 12}"
EMBEDDING_DIMS="${EMBEDDING_DIMS:-3 8 16 64}"
SWEEP_MAX_ITERS="${SWEEP_MAX_ITERS:-10000}"
SWEEP_SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL:-100}"
RUNS_DIR="report/threejs/digits-3d/runs"

mkdir -p "${RUNS_DIR}"

for embedding_dim in ${EMBEDDING_DIMS}; do
  for num_digits in ${DIGIT_COUNTS}; do
    for num_letters in ${LETTER_COUNTS}; do
      for mode in unconstrained sqrt_dim unit; do
        case "${mode}" in
          unconstrained) fixed=false; radius="" ;;
          sqrt_dim) fixed=true; radius="" ;;
          unit) fixed=true; radius=1 ;;
        esac

        name="dim-${embedding_dim}_digits-${num_digits}_letters-${num_letters}_${mode}"
        echo "=== ${name} ==="
        NUM_DIGITS="${num_digits}" \
        NUM_LETTERS="${num_letters}" \
        EMBEDDING_DIM="${embedding_dim}" \
        WTE_FIXED_NORM="${fixed}" \
        WTE_FIXED_NORM_VALUE="${radius}" \
        MAX_ITERS="${SWEEP_MAX_ITERS}" \
        SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL}" \
        OUT_DIR="out/digits_3d_sweep/${name}" \
        TRAJECTORY_FILE="${RUNS_DIR}/${name}.json" \
          bash demos/digits_3d_trajectory_demo.sh
      done
    done
  done
done

cat <<EOF
Sweep complete. Serve the repository with:
  python3 -m http.server 8000

Example result:
  http://localhost:8000/report/threejs/digits-3d/index.html?data=runs/dim-16_digits-10_letters-4_sqrt_dim.json
EOF
