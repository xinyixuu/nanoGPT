#!/usr/bin/env bash
# run_dataset_colorization.sh

# Defaults + positional overrides
OUT_DIR=${1:-out}
DATASET=${2:-shakespeare_char}
MODE=${3:-minmax}
NUM_TOKENS=${4:-1024}
BLOCK_SIZE=${5:-256}
SPLIT=${6:-train}
OUTPUT_FILE=${7:-val_colour.txt}
DEVICE=${8:-cuda:0}

python colorize_dataset.py \
  --out_dir     "$OUT_DIR" \
  --dataset     "$DATASET" \
  --split       "$SPLIT" \
  --num_tokens  "$NUM_TOKENS" \
  --device      "$DEVICE" \
  --block_size  "$BLOCK_SIZE" \
  --mode        "$MODE" \
  --output_file "$OUTPUT_FILE"

