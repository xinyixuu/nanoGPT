#!/bin/bash
# demos/run_recurrent_training.sh
python train_recurrent.py \
    --resume_ckpt out/ckpt.pt \
    --dataset cosmopedia_100k \
    --block_size 256 \
    --latent_steps 128


