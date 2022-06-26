#!/usr/bin/env python3
########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import sys

import numpy as np

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

MODE = 'jsonl' # txt jsonl
CUTOFF = 64*1024 # minimum data size

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} file1.{MODE} file2.{MODE} file3.{MODE} ... > train.bin', file=sys.stderr)
    print('Resulting .bin files can be concatenated or streamed.', file=sys.stderr)
    print('The filename "-" can be used for standard input.', file=sys.stderr)
    sys.exit(-1)

file_objs = (
    print(f'Tokenizing {input_file}', file=sys.stderr) or
        (open(input_file) if input_file != '-' else sys.stdin)
    for input_file in sys.argv[1:]
)

if MODE == 'txt':
    data_raws = (file_obj.read() for file_obj in file_objs)
elif MODE == 'jsonl':
    import json
    data_raws = (json.loads(line)['text'] for file_obj in file_objs for line in file_obj)

skip = 0
for data_raw in data_raws:
    if len(data_raw) < CUTOFF * 4:
        skip += 1
        continue
    data_code = tokenizer.encode(data_raw)
    if len(data_code) < CUTOFF:
        skip += 1
        continue

    if skip > 0:
        print(f'Skipped {skip} short entries.', file=sys.stderr)

    print(f'Raw length = {len(data_raw)}', file=sys.stderr)
    print(f'Tokenized length = {len(data_code)}', file=sys.stderr)
    
    out = np.array(data_code, dtype='uint16').tobytes()
    print(f'Bytes length = {len(out)}', file=sys.stderr)
    sys.stdout.buffer.write(len(out).to_bytes(8, 'little'))
    sys.stdout.buffer.write(out)
