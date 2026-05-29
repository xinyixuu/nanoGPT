#!/bin/bash

### Instructions:
# 1. Replace "INSERT_URL_WITH_FILES" with the actual URL to the Parquet files.
# 2. Modify the "include_keys" array to specify the keys you want to include in the output.
# 3. (Optionally) Modify the "value_prefixes" array to set prefixes for each value, use "" for empty prefixes
# 4. Set "--skip_empty" to true if you want to skip empty fields, or false if not needed.
# 5. Set "--no_output_text" to true if you plan to process the intermediate json files in a custom manner.

# Run the Python script with the specified arguments

# Add url with dataset here:
train_url="https://huggingface.co/datasets/rhyliieee/tagalog-filipino-english-translation/resolve/main/train_data.csv?download=true"
test_url="https://huggingface.co/datasets/rhyliieee/tagalog-filipino-english-translation/resolve/main/test_data.csv?download=true"

train_filename="train.txt"

if [ -f "$train_filename" ]; then
    echo "$train_filename already exists. Skipping download."
else
    wget -O "$train_filename" "$train_url"
fi

test_filename="test.txt"

if [ -f "$test_filename" ]; then
    echo "$test_filename already exists. Skipping download."
else
    wget -O "$test_filename" "$test_url"
fi

cat "$train_filename" "$test_filename" > "input.txt"

python3 prepare.py --method char -t input.txt

