#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# Using intermiediate datasets for tokenization
pushd "data/snac_cvja"
bash get_ipa.sh
popd

pushd "data/snac_cvko"
bash get_ipa.sh
popd

pushd "data/snac_cvzh"
bash get_ipa.sh
popd

# running the training
# python3 optimization_and_search/run_experiments.py --config explorations/multidataset.json --output_dir out_multi_zh

# using the intermediate datasets for training
python3 train.py \
    --dataset snac_cvzh \
    --dataset_list snac_cvzh snac_cvko snac_cvja \
    --dataset_sampling_probs 1 1 1 \
    --use_lsv \
    --max_iters 2500 \
    --apply_lsv_at_layer_idx 0 \
    --eval_interval 500 \
    --eval_iters 50 \
    --dataset_interleaving \
    --dataset_interleaving_shuffle \
    --lsv_variant one_hot \
    --out_dir "out_muti_zh" \
    --init_from "scratch" \
    --gns_type exact
popd