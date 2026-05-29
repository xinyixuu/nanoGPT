#!/bin/bash
# create_new_exploration.sh

new_exploration_name="$1"

cp ./explorations/sample.yaml "./explorations/${new_exploration_name}.yaml"
sed -i "s/sample.yaml/${1}.yaml/g" "./explorations/${new_exploration_name}.yaml"
