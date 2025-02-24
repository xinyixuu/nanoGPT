#!/bin/bash

set -x

dog_image_url="https://huggingface.co/nickmuchi/vit-finetuned-cats-dogs/resolve/main/images/dog.jpg"
cat_image_url="https://huggingface.co/nickmuchi/vit-finetuned-cats-dogs/resolve/main/images/cat.jpg"

echo "cat prompt instead" | cowsay

python script.py \
    --image_path "$cat_image_url" \
    --save_npy cat_emb.npy

python script.py \
    --image_path "$dog_image_url" \
    --load_npy cat_emb.npy

echo "dog prompt instead" | cowsay

python script.py \
    --image_path "$dog_image_url" \
    --save_npy dog_emb.npy

python script.py \
    --image_path "$cat_image_url" \
    --load_npy dog_emb.npy
