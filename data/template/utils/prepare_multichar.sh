#!/bin/bash
# data/template/utils/prepare_multichar.sh

set +x

arr=(cvp part_of_speech in_word_position since_newline)

for i in ${arr[@]}; do
  echo "beginning $i"
  # create copy of file
  filename="input_${i}.txt"
  cp input.txt "$filename"
  # convert chars
  python3 ./utils/char_convert.py "$filename" --method "$i"
  # create meta.pkl dictionary, but skip creation of bin
  python3 prepare.py -t "tokensfile.txt" --method char --skip_tokenization
  # tokenize, creating bins and meta.pkl files
  python3 prepare.py -t "$filename" --method char --reuse_char
  # move to dir
  dirname="mc_$i"
  mkdir -p "$dirname"
  mv "$filename" "$dirname"
  mv *.bin "$dirname"
  mv meta.pkl "$dirname"
  mv tokensfile.txt "$dirname"
done

