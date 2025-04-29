#!/bin/bash

# Add url with dataset here:
url="https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/tree/main/kor_Hang/"

# uncomment and fill in if url has parquet datasets
python3 ./utils/get_parquet_dataset.py \
  --url "${url}" \
  --include_keys "text" \
  --value_prefix $'\n#T: '
