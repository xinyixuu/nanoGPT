#!/bin/bash
# Note: may need to run this script as `source start_tensorboard.sh` to utilize environment

# Default Port
PORT=6006

# Check if user provided a specific port
if [ ! -z "$1" ]; then
  PORT="$1"
fi

tensorboard --logdir=./logs --port="$PORT" --bind_all | python3 -m tensorboard.main --logdir=./logs --port="$PORT" --bind_all
