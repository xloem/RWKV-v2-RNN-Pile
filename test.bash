#!/usr/bin/env bash
for WEIGHTS_FILE in *.pth
do
    WEIGHTS="${WEIGHTS_FILE%.pth}"
    for model in GPT_RNN/"$WEIGHTS" GPT_FULL/"$WEIGHTS"
    do
        echo "$model"
        python3 main.py --task lambada --model "$model" --device cuda --batch_size 16
    done
done
