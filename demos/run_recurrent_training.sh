#!/bin/bash
# demos/run_recurrent_training.sh
# python train_recurrent.py \
#     --resume_ckpt out/ckpt.pt \
#     --dataset cosmopedia_100k \
#     --block_size 256 \
#     --latent_steps 128

# python train_recurrent.py \
#     --resume_ckpt out/ckpt.pt \
#     --latent_steps 128 \
#     --skip_steps 16 \
#     --optimizer lookahead \
#     --lookahead_inner_opt adafactor \
#     --tensorboard_log

# python train_recurrent.py \
#   --resume_ckpt out/ckpt.pt \
#   --latent_steps 16 \
#   --skip_steps 16 \
#   --optimizer lookahead \
#   --lookahead_inner_opt adafactor \
#   --tensorboard_log

python train_recurrent.py \
  --resume_ckpt out/ckpt.pt \
  --latent_steps 16 --skip_steps 16 \
  --optimizer lookahead --lookahead_inner_opt adafactor \
  --reset_optim  --tensorboard_log --eval_interval 10 --eval_iters 20

