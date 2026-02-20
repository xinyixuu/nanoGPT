#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

IPA_file="ipa_zho_Hans.txt"
Text_file="text_zho_Hans.txt"

python3 "$script_dir"/utils/text_extractor.py "input.txt" "$Text_file" "ZH"

python3 "$script_dir"/utils/zh_to_ipa.py text_zho_Hans.txt ipa_text_zho_Hans.txt --input_type text --no-wrapper --stats_json zh_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_zho_Hans.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt