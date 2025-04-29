#!/bin/bash
# run_hp_search.sh

python hyperparam_search.py \
  --orig_settings baseline.yaml \
  --param_names n_layer n_head n_embd mlp_size\
  --increments 1 1 16 16\
  --random_iterations 5 \
  --iterations 1 \
  --num_iterations 20 \
  --results_file random_lhem_out.yaml

