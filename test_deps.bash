#!/usr/bin/env bash
WEIGHTS=20220615-10803
# running these in parallel for efficiency
{
    pip3 install git+https://github.com/xloem/lm-evaluation-harness@string-names
} &
{
    wget -c https://github.com/BlinkDL/RWKV-v2-RNN-Pile/releases/download/"$WEIGHTS"/"$WEIGHTS".zip
    unzip "$WEIGHTS".zip
    #rm "$WEIGHTS".zip
} &
{
    git clone https://github.com/BlinkDL/RWKV-v2-RNN-Pile
    cd RWKV-v2-RNN-Pile
    ln -sf ../*.pth .
    ln -sf ../main.py
} &
wait
