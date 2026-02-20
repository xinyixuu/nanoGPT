#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

Text_file="text_kor_Hang.txt"

python3 "$script_dir"/utils/text_extractor.py "input.txt" "$Text_file" "KO"

python3 "$script_dir"/prepare.py -t "text_kor_Hang.txt" --method tiktoken