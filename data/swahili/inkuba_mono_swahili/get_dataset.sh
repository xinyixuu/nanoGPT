#!/bin/bash

# Add url with dataset here:
url="https://huggingface.co/datasets/Alfaxad/Inkuba-Mono-Swahili/resolve/main/sw/data.txt?download=true"

filename="input.txt"

if [ -f "$filename" ]; then
    echo "$filename already exists. Skipping download."
else
    wget -O "$filename" "$url"
fi

bash ./utils/large_file_prepare.sh input.txt char
