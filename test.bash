#!/usr/bin/env bash
for WEIGHTS_FILE in *.pth
do
    WEIGHTS="${WEIGHTS_FILE%.pth}"
    echo "CLASSIC"
    pushd RWKV-v2-RNN-Pile
    python3 main.py --task lambada --model GPT_FULL/"$WEIGHTS" --device cuda --batch_size 16
    popd
    
    for model in GPT_RNN/"$WEIGHTS" GPT_FULL/"$WEIGHTS"
    do
        for device in cuda cpu
        do
            echo "$model-$device"
            python3 main.py --task lambada --model "$model" --device "$device" --batch_size 16
        done
    done
done
