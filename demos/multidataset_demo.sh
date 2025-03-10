#!/bin/bash

# Get current script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# get into main directory
pushd "$script_dir/../" > /dev/null

# tokenize on commonvoice_zh
bash data/commonvoice_zh/get_dataset.sh

# tokenize on commonvoice_ko
bash data/commonvoice_ko/get_dataset.sh

# tokenize on commonvoice_ja
bash data/commonvoice_ja/get_dataset.sh

# running the training
python3 optimization_and_search/run_experiments.py --config explorations/multidataset.json --output_dir out_multi_zh
popd
