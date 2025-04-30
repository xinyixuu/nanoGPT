#!/bin/bash

data_dir="data"

if [[ ! -d "$data_dir" ]]; then
    mkdir -p "$data_dir"
fi

pushd "$data_dir"
curl -LO https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/kokoro-speech-v1_3.zip
unzip kokoro-speech-v1_3.zip
curl -LO http://archive.org/download/gongitsune_um_librivox/gongitsune_um_librivox_64kb_mp3.zip
curl -LO http://www.archive.org/download/caucasus_no_hagetaka_um_librivox/caucasus_no_hagetaka_um_librivox_64kb_mp3.zip
unzip gongitsune_um_librivox_64kb_mp3.zip -d gongitsune-by-nankichi-niimi
unzip caucasus_no_hagetaka_um_librivox_64kb_mp3.zip -d caucasus-no-hagetaka-by-yoshio-toyoshima
popd

python3 extract.py --size tiny
