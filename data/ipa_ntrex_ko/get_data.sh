#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

IPA_file="ipa_kor_Hang.txt"
Text_file="text_kor_Hang.txt"

python3 "$script_dir"/utils/text_extractor.py "input.txt" "$Text_file" "KO"

python3 "$script_dir"/utils/ko_en_to_ipa.py text_kor_Hang.txt --text_input --text_output ipa_text_kor_Hang.txt --stats_json ko_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_kor_Hang.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt