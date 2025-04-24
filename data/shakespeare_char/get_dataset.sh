#!/usr/bin/env bash

# Download the tiny Shakespeare dataset from GitHub
wget -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python3 prepare.py --method char -t input.txt

