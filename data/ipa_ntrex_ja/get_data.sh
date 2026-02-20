#!/bin/bash

# Downloads the parquet shards from Hugging Face and emits text_eng* columns to input.txt
# You can modify INCLUDE_KEYS to pull different language columns from the schema.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

IPA_file="ipa_jpn_Jpan.txt"
Text_file="text_jpn_Jpan.txt"

python3 "$script_dir"/utils/text_extractor.py "input.txt" "$Text_file" "JA"

python3 "$script_dir"/utils/ja2ipa.py text_jpn_Jpan.txt ipa_text_jpn_Jpan.txt --text_output --use_spacy --text_no_sentence --stats_json ja_stats.json

python3 "$script_dir"/prepare.py -t "ipa_text_jpn_Jpan.txt" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt