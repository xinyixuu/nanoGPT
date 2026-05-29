#!/bin/bash

# Add url with dataset here:
url="https://huggingface.co/datasets/michsethowusu/english-swahili_sentence-pairs/resolve/main/English-Swahili_Sentence-Pairs.csv?download=true"

filename="input.txt"

if [ -f "$filename" ]; then
    echo "$filename already exists. Skipping download."
else
    wget -O "$filename" "$url"
fi

python3 prepare.py -t input.txt --method char
