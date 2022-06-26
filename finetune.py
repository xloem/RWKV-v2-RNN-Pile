########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging
import datetime
import json
import sys
from src.model_train import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
import torch
from torch.utils.data import Dataset
import numpy as np
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# lightweight values to test on my 2GB gpu
ctx_len = 96
n_layer = 4#3
n_embd = 256#128

#ctx_len = 768
#n_layer = 24
#n_embd = 1024
vocab_size = 50277

model_type = 'RWKV'
model_name = '20220524-4006'

datafile = 'data.bin'#'train.npy' # use 'prepare-data.py' to tokenize .txt into .npy

########################################################################################################

batch_size = 8 
# The batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in src/model_train.py.
# you can reduce B_GROUP_FORWARD and B_GROUP_BACKWARD to make it easier to find a good batch_size for your GPU.
# just remember B_GROUP_FORWARD=8 and B_GROUP_BACKWARD=2 is the fastest.

lr_init = 2e-5
lr_final = 1e-5

n_epoch = 100 # the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
epoch_length_fixed = 10000

epoch_save_frequency = 5 # 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_path = 'trained-'

########################################################################################################

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 1e-8

ctx_len -= 1 # to hold recurrence k & kv

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

class Dataset(Dataset):
    def __init__(self, datafile, vocab_size, ctx_len, group_size, batch_size):
        self.datafile = open(datafile, 'rb') if datafile != '-' else sys.stdin.buffer
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.group_size = group_size
        self.batch_size = batch_size
        self.count = 0
        self.data = [self._new_offset_idx_data() for group_idx in range(self.group_size)]

    def _new_offset_idx_data(self):
        byte_size = int.from_bytes(self.datafile.read(8), 'little')
        assert byte_size > 0
        print('reading %d bytes' % byte_size)
        data = np.frombuffer(self.datafile.read(byte_size), dtype='uint16').astype('int')
        data_size = len(data)
        self.count += 1
        print('data %d has %d tokens, %d unique.' % (self.count, data_size, self.vocab_size))
        return 0, self.count - 1, data

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return sum(len(data) for offset, idx, data in self.data) // self.ctx_len

    def __getitem__(self, _idx):
        # cheat: pick random data from among the group
        group_idcs = np.random.choice(self.group_size, self.batch_size, replace=False)

        recurrences = []
        dixes = []
        for group_idx in group_idcs:
            offset, idx, data = self.data[group_idx]
            dixes.append(data[offset : offset + self.ctx_len+1])
            if offset + ctx_len * 2 <= len(data):
                self.data[group_idx] = (offset + ctx_len, idx + 1, data)
                recurrences.append((idx, True))
            else:
                self.data[group_idx] = self._new_offset_idx_data()
                recurrences.append((idx, False))
        
        #epoch_offsets = self.epoch_offsets + self.ctx_len * idx
        #dixes = [
        #    self.data[epoch_offset:epoch_offset + self.ctx_len+1]
        #    for epoch_offset in epoch_offsets
        #]
        x = torch.stack([
            torch.tensor(dix[:-1], dtype=torch.long,
                         device=torch.device('cuda'))
            for dix in dixes
        ])
        y = torch.stack([
            torch.tensor(dix[1:], dtype=torch.long,
                         device=torch.device('cuda'))
            for dix in dixes
        ])
        return recurrences, x, y

print('loading data... ' + datafile)
train_dataset = Dataset(datafile, vocab_size, ctx_len, batch_size * 2, batch_size)

########################################################################################################
# Train model
########################################################################################################

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    try:
        print('loading ' + model_name)
        m2 = torch.load(model_name + '.pth')
        model.load_state_dict(m2)
    except:
        print('failed to load, will make new')

    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, None, tconf)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
