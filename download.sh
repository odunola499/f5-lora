#!/bin/bash

# Download the files
wget "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors?download=true"
wget "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/vocab.txt?download=true"

# Rename files by removing '?download=true'
for f in *\?download=true; do
    mv "$f" "${f%\?download=true}"
done

echo "Downloads complete and filenames cleaned."
