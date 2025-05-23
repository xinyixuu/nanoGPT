#!/bin/bash
# data/template/utils/prepare_multichar.sh

set +x

arr=(cvp part_of_speech in_word_position since_newline)

for i in ${arr[@]}; do
  echo "begginning $i"
  # create copy of file
  filename="input_${i}.txt"
  cp input.txt "$filename"
  # convert chars
  python3 ./utils/char_convert.py "$filename" --method "$i"
  # tokenize, creating bins and meta.pkl files
  python3 prepare.py -t "$filename" --method char
  # move to dir
  dirname="mc_$i"
  mkdir -p "$dirname"
  mv "$filename" "$dirname"
  mv *.bin "$dirname"
  mv meta.pkl "$dirname"
done

