#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

Text_file="text_jpn_Jpan.txt"

python3 "$script_dir"/utils/text_extractor.py "input.txt" "$Text_file"  "JA"

python3 "$script_dir"/prepare.py -t "text_jpn_Jpan.txt" --method tiktoken