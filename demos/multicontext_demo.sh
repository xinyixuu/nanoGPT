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
   --max_iters 3000 \
   --dropout 0.2 \
   --save_major_ckpt_interval 250 \
   --use_rotary_embeddings \
   --compile

python3 sample.py \
--out_dir ./out \
--multicontext \
--multicontext_datasets \
    shakespeare_char \
    shakespeare_char_mobius \
    shakespeare_char_pos_2 \
    shakespeare_char_punct \
    shakespeare_char_order \
--multicontext_start \
    "Second " \
    "1234567" \
    "aaaaaa " \
    "212122_" \
    "123456_" \
--max_new_tokens \
    256 \
--top_k \
    1 10 \
--num_samples 1
python3 sample.py --out_dir ./out --multicontext --multicontext_datasets  shakespeare_char shakespeare_char_mobius shakespeare_char_pos_2 shakespeare_char_punct shakespeare_char_order  --multicontext_start  "Second " "1234567" "aaaaaa " "212122_" "123456_" --max_new_tokens 512 --top_k 1 --num_samples 1
