#!/bin/bash
# multicontext_demo.sh

python3 train.py \
  --training_mode multicontext \
  --multicontext \
  --multicontext_datasets \
      ./data/shakespeare_char \
      ./data/shakespeare_char_mobius \
      ./data/shakespeare_char_pos_2 \
      ./data/shakespeare_char_punct \
      ./data/shakespeare_char_order \
   --vocab_sizes 65 256 20 4 24 \
   --max_iters 6000 \
   --save_major_ckpt_interval 250 \
   --use_rotary_embeddings
