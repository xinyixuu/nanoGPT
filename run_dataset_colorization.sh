#!/bin/bash
# run_dataset_colorization.sh

python colorize_dataset.py \
  --out_dir        out \
  --dataset        filipino/tagalog_filipino_eng_translation  \
  --split          val \
  --num_tokens     2048 \
  --device         cuda:0 \
  --block_size     256 \
  --output_file    val_colour.txt
