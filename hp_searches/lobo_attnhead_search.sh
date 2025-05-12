#!/bin/bash
# lobo_attnhead_search.sh

python3 hyperparam_search.py \
  --orig_settings ./hp_searches/lobo_attnhead_search.yaml \
  --param_names \
        n_layer \
        n_head \
        n_embd \
        mlp_size \
        n_qk_head_dim \
        n_v_head_dim \
        flash_lobo_log_const \
  --increments \
        1 \
        1 \
        16 \
        16 \
        16 \
        16 \
        0.1 \
  --random_iterations 3 \
  --iterations 1 \
  --num_iterations 2000 \
  --results_file results.yaml

