########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, json, time, types, copy, sys, os
import torch
from torch.nn import functional as F
import torch.nn as nn

from transformers import PreTrainedTokenizerFast

# RUN_DEVICE = 'cpu' # cpu cuda
# ctx_len = 768
# n_layer = 12
# n_embd = 768

RUN_DEVICE = 'cpu'
ctx_len = 768
n_layer = 24
n_embd = 1024

MODEL_NAME = '20220524-4006'

vocab_size = 50277
VOCAB_NAME = '20B_tokenizer.json'

print(f'\n* running on {RUN_DEVICE}')

################################################################################################################

class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_mix = nn.Parameter(torch.ones(1, 1, n_embd, device=RUN_DEVICE))

        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False, device=RUN_DEVICE)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False, device=RUN_DEVICE)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False, device=RUN_DEVICE)

    def time_shift(self, x):
        return torch.cat([self.xx.expand(x.shape[0],1,self.xx.shape[-1]), x], dim=-2)[...,:-1,:]
        #return torch.cat([self.xx[...,None,:], x], dim=-2)[...,:-1,:]

    def forward(self, x):
        xx = x[...,-1,:].detach()
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        self.xx = xx

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(x)) * kv
        return rkv

class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(n_embd, 1, device=RUN_DEVICE))
        self.time_curve = torch.tensor([-(ctx_len - 2 - i) for i in range(ctx_len-1)], device=RUN_DEVICE).unsqueeze(0)
        self.time_first = nn.Parameter(torch.ones(n_embd, 1, device=RUN_DEVICE) * math.log(0.3))
        
        self.time_mix = nn.Parameter(torch.ones(1,1,n_embd, device=RUN_DEVICE))

        self.key = nn.Linear(n_embd, n_embd, bias=False, device=RUN_DEVICE)
        self.value = nn.Linear(n_embd, n_embd, bias=False, device=RUN_DEVICE)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False, device=RUN_DEVICE)

        self.output = nn.Linear(n_embd, n_embd, bias=False, device=RUN_DEVICE)

    def time_shift(self, x):
        return torch.cat([self.xx.expand(x.shape[0],1,self.xx.shape[-1]), x], dim=-2)[...,:-1,:]

    def forward(self, x):
        B, T, C = x.size()

        xx = x[...,-1,:].detach()
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)

        k = self.key(x).transpose(-1, -2)
        v = self.value(x).transpose(-1, -2)
        r = self.receptance(x)

        mm = torch.max(k, dim=-1).values

        # giving mm a mantissa of 1 prevents floating point changes
        mm = torch.frexp(mm)
        mm = torch.ldexp(torch.tensor(0.5,device=k.device), mm.exponent)

        k = torch.exp(k - mm[...,None])
        #import pdb; pdb.set_trace()

        kv = k * v

        self.time_w = torch.cat([torch.exp(self.time_decay) * self.time_curve.to(self.time_decay.device), self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        recurrent_scaling_factor = torch.exp(self.mm - mm)
        k = torch.cat([(self.bb * recurrent_scaling_factor)[...,None],k], dim=-1)
        kv = torch.cat([(self.aa * recurrent_scaling_factor)[...,None],kv], dim=-1)
        T += 1
        extra_decay = w[...,-3]

        w = w[:,-T:].unsqueeze(1)
        wkv = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(kv), w, groups=C)[...,1:]
        wk = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w, groups=C)[...,1:] + 1e-9

        self.xx = xx
        self.bb = (wk[...,-1] - w[...,0,-1] * k[...,-1]) * extra_decay + k[...,-1]
        self.aa = (wkv[...,-1] - w[...,0,-1] * kv[...,-1]) * extra_decay + kv[...,-1]
        self.mm = mm

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        
        rwkv = self.output(rwkv)
        return rwkv

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd, device=RUN_DEVICE)
        self.ln2 = nn.LayerNorm(n_embd, device=RUN_DEVICE)
        
        self.att = RWKV_TimeMix(layer_id)
        self.ffn = RWKV_ChannelMix(layer_id)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x

class RWKV_GPT(nn.Module):
    def __init__(self, MODEL_NAME=MODEL_NAME, recurrent=True):
        super().__init__()
        print('\nloading RWKV-GPT', MODEL_NAME)

        self.recurrent = recurrent

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)
        self.emb = nn.Embedding(vocab_size, n_embd, device=RUN_DEVICE)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd, device=RUN_DEVICE)
        self.head = nn.Linear(n_embd, vocab_size, bias=False, device=RUN_DEVICE)

        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE)))
        self.clear()
        self.eval()

    def clear(self, *batch_idcs):
        zeros = torch.zeros(1, n_embd, device=RUN_DEVICE)
        for block in self.blocks:
            if not batch_idcs:
                block.ffn.xx = zeros
                block.att.xx = zeros
                block.att.aa = zeros
                block.att.bb = zeros
                block.att.mm = zeros
            else:
                for idx in batch_idcs:
                    block.ffn.xx[idx] = zeros[0]
                    block.att.xx[idx] = zeros[0]
                    block.att.aa[idx] = zeros[0]
                    block.att.bb[idx] = zeros[0]
                    block.att.mm[idx] = zeros[0]
    def save(self, *targets):
        for target in targets:
            target.xx = {}
            target.aa = {}
            target.bb = {}
            target.mm = {}
        for idx, block in enumerate(self.blocks):
            for batch, target in enumerate(targets):
                target.xx[f'ffn.{idx}'] = block.ffn.xx[batch]
                target.xx[f'att.{idx}'] = block.att.xx[batch]
                target.aa[f'att.{idx}'] = block.att.aa[batch]
                target.bb[f'att.{idx}'] = block.att.bb[batch]
                target.mm[f'att.{idx}'] = block.att.mm[batch]
    def load(self, *targets):
        zeros = torch.zeros(1, n_embd, device=RUN_DEVICE)
        for idx, block in enumerate(self.blocks):
            block.ffn.xx = torch.stack([target.xx[f'ffn.{idx}'] for target in targets]).to(RUN_DEVICE)
            block.att.xx = torch.stack([target.xx[f'att.{idx}'] for target in targets]).to(RUN_DEVICE)
            block.att.aa = torch.stack([target.aa[f'att.{idx}'] for target in targets]).to(RUN_DEVICE)
            block.att.bb = torch.stack([target.bb[f'att.{idx}'] for target in targets]).to(RUN_DEVICE)
            block.att.mm = torch.stack([target.mm[f'att.{idx}'] if hasattr(target, 'mm') else zeros[0] for target in targets]).to(RUN_DEVICE)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        if not self.recurrent:
            self.clear()
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)

        return x

################################################################################################################

time_buf = {}

class RWKV_RNN():
    def __init__(self, MODEL_NAME=MODEL_NAME):
        print('\nloading RWKV-RNN', MODEL_NAME)
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)

        self.w = types.SimpleNamespace()
        
        w = torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))

        for x in w.keys():
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x]))
            if '.time_first' in x:
                w[x] = torch.exp(w[x])
                    
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])
    
        self.clear()
    
    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.mm = {}
    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.mm = copy.deepcopy(self.mm)
    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        if hasattr(target, 'mm'):
            self.mm = copy.deepcopy(target.mm)
        else:
            self.mm = {}

    def LN(self, xx, w):
        return F.layer_norm(xx, (n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)

        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ x)
        k = torch.square(torch.relu(w.key.weight @ x))
        kv = w.value.weight @ k

        return r * kv

    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)
            self.aa[name] = torch.zeros(n_embd, device=RUN_DEVICE)
            self.bb[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        if name not in self.mm:
            self.mm[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)
        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ x)

        k = w.key.weight @ x
        mm = k

        # giving mm a mantissa of 1 prevents floating point changes
        mm = torch.frexp(mm)
        mm = torch.ldexp(torch.tensor(0.5,device=k.device), mm.exponent)
        
        k = torch.exp(k - mm)
        v = w.value.weight @ x
        kv = k * v

        recurrent_scaling_factor = torch.exp(self.mm[name] - mm)
        aa = self.aa[name] * recurrent_scaling_factor
        bb = self.bb[name] * recurrent_scaling_factor
        self.aa[name] = w.time_decay * aa + kv
        self.bb[name] = w.time_decay * bb + k
        a = aa + w.time_first * kv
        b = bb + w.time_first * k
        self.mm[name] = mm

        rwkv = r * a / (b + 1e-9)

        return w.output.weight @ rwkv

    def run(self, ctx):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        for i in range(n_layer):
            x = self.LN(x, w.blocks[i].ln1)
            x = x + self.SA(x, w.blocks[i].att, f'att.{i}')
            x = self.LN(x, w.blocks[i].ln2)
            x = x + self.FF(x, w.blocks[i].ffn, f'ffn.{i}')

        x = self.LN(x, w.ln_out)

        x = w.head.weight @ x
        x = x.tolist()

        return x

################################################################################################################
