#!/bin/bash
# create_new_exploration.sh

new_exploration_name="$1"

cp ./explorations/sample.yaml "./explorations/${new_exploration_name}.yaml"
