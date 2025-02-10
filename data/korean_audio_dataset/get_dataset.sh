#!/bin/bash

url="https://huggingface.co/datasets/JaepaX/korean_dataset/tree/main/data"

 python utils/get_parquet_to_json.py \
 --url "${url}" \
 --range_start 0 \
 --range_end 1 \
 --include_keys transcription \
 --output_json ko.json

 # Run program to convert tsv into json format.
output_file="ko.json"

# Run program to convert sentences into IPA format.
echo "Converting sentences to IPA..."
python3 ./utils/ko_en_to_ipa.py "$output_file" --input_json_key "transcription" --output_json_key "sentence_ipa"

output_ipa="ko_ipa.txt"
echo "export IPA to txt file"
python3 ./utils/extract_json_values.py "$output_file" "sentence_ipa" "$output_ipa"

echo "IPA conversion finished."

output_ipa_cleaned="${output_ipa%%.txt}_clean.txt"
# Clean any square brackets in this version
sed -E 's/\[{5}([^][]+)]{5}/\1/g' "${output_ipa}"> "${output_ipa_cleaned}"

# Tokenization step to create train.bin and val.bin files.
python3 prepare.py -t "${output_ipa_cleaned}" --method custom_char_byte_fallback --custom_chars_file ../template/phoneme_list.txt

