########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from importlib import abc # works around python bug importing timex
from torch.utils.cpp_extension import load
import math
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

##try:
##    raise Exception()
#import fewbit
##except:
Linear = nn.Linear
relu = F.relu
sigmoid = F.sigmoid
##else:
#Linear = fewbit.LinearGRP
#relu = fewbit.functional.relu
#sigmoid = fewbit.functional.sigmoid

#def stable_linear(linear, operand):
#    return torch.stack([
#        linear(operand[...,idx,:]) for idx in range(operand.shape[-2])
#    ], dim=-2)
#def stable_exp(operand):
#    return torch.stack([
#        torch.exp(operand[...,idx]) for idx in range(operand.shape[-1])
#    ], dim=-1)
#def stable_conv1d(data, kernel, groups='ignored'):
#    data_size = data.shape[-1]
#    kernel_size = kernel.shape[-1]
#    return torch.stack([
#        (data[...,idx:idx+kernel_size-1] * kernel.transpose(-2,-3)[...,:-1]).sum(dim=-1) + data[...,idx+kernel_size-1] * kernel.transpose(-2,-3)[...,-1]
#        for idx in range(0,data_size-kernel_size+1)
#    ], dim=-1)

########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = 1024          # increase this if your ctx_len > 1024
B_GROUP_FORWARD = 1   # set to 8 for best performance
B_GROUP_BACKWARD = 1  # set to 2 for best performance

timex_cuda = load(name="timex", sources=["cuda/timex_op.cpp", "cuda/timex_cuda.cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'])


class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.forward(w, k, wk, eps, B, C, T)
        return wk

    @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        return (gw.sum(dim=0), gk, None, None, None, None)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

RWKV_K_CLAMP = 60  # e^60 = 1e26
RWKV_K_EPS = 1e-9
RWKV_HEAD_QK_DIM = 256

# def RWKV_Init(module, config):  # fancy initialization of all lin & emb layer in the module
#     for m in module.modules():
#         if not isinstance(m, (Linear, nn.Embedding)):
#             continue
#         with torch.no_grad():
#             name = '[unknown weight]'
#             for name, parameter in module.named_parameters():  # find the name of the weight
#                 if id(m.weight) == id(parameter):
#                     break

#             shape = m.weight.data.shape
#             gain = 1.0
#             scale = 1.0  # extra scale for gain

#             if isinstance(m, nn.Embedding):
#                 gain = math.sqrt(max(shape[0], shape[1]))
#                 if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # token emb?
#                     scale = 1e-4
#                 else:
#                     scale = 0

#             if isinstance(m, Linear):
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                 if shape[0] > shape[1]:
#                     gain = math.sqrt(shape[0] / shape[1])
#                 if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # final projection?
#                     scale = 0.5

#             if hasattr(m, 'scale_init'):
#                 scale = m.scale_init

#             # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

#             gain *= scale
#             if scale == -999:
#                 nn.init.eye_(m.weight)
#             elif gain == 0:
#                 # zero init is great for some RWKV matrices
#                 nn.init.zeros_(m.weight)
#             elif gain > 0:
#                 nn.init.orthogonal_(m.weight, gain=gain)
#             else:
#                 nn.init.normal_(m.weight, mean=0.0, std=-scale)


class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len + 1
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        ############# fancy init of time_w curves ###################################
        f1_begin = 3.0
        f1_end = 1.2
        f2_begin = 0.65
        f2_end = 0.4
        with torch.no_grad():  # initial time_w curves for better convergence
            decay_speed = torch.ones(attn_sz, 1)
            first_sa_layer_id = 1
            for h in range(attn_sz):
                f1 = f1_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
                f2 = f2_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
                if layer_id == first_sa_layer_id:
                    f1 += 0.5
                if layer_id == config.n_layer-2:
                    f2 = 0.4
                if layer_id == config.n_layer-1:
                    f2 = 0.37
                decay_speed[h][0] = math.pow(f2, h / (attn_sz-1) * 7) * f1
        self.time_decay = nn.Parameter(torch.log(decay_speed)) # will use exp(self.time_decay) to ensure time_decay > 0
        self.time_curve = torch.tensor(
            [-(self.ctx_len - 2 - i) for i in range(self.ctx_len-1)]).unsqueeze(0)
        self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3))
        #############################################################################

        with torch.no_grad():  # init to "shift half of the channels"
            ww = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix = nn.Parameter(ww)

        self.key = Linear(config.n_embd, attn_sz, bias=False)
        self.value = Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = Linear(config.n_embd, attn_sz, bias=False)

        self.output = Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def time_shift(self, x):
        return torch.cat([self.xx.expand(x.shape[0], x.shape[-1])[...,None,:], x], dim=-2)[...,:-1,:]

    def forward(self, x):
        B, T, C = x.size()
        assert T == self.ctx_len - 1

        xx = x[...,-1,:]
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)

        k = self.key(x).transpose(-1, -2)
        v = self.value(x).transpose(-1, -2)
        r = self.receptance(x)
        #k = stable_linear(self.key, x).transpose(-1, -2)
        #v = stable_linear(self.value, x).transpose(-1, -2)
        #r = stable_linear(self.receptance, x)

        mm = torch.max(k, dim=-1).values

        # giving mm a mantissa of 1 prevents floating point changes
        mm = torch.frexp(mm)
        mm = torch.ldexp(torch.tensor(0.5,device=k.device), mm.exponent)

        k = torch.exp(k - mm[...,None])
        #k = stable_exp(k - mm[...,None])
        kv = k * v

        self.time_curve = self.time_curve.to(x.device)

        self.time_w = torch.cat(
            [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        recurrent_scaling_factor = torch.exp(self.mm - mm)
        k = torch.cat([(self.bb * recurrent_scaling_factor)[...,None],k], dim=-1)
        kv = torch.cat([(self.aa * recurrent_scaling_factor)[...,None],kv], dim=-1)
        T += 1
        extra_decay = w[...,-3]

        wkv = TimeX.apply(w.cuda(), kv.cuda(), B, C, T, 0)[...,1:].to(x.device)
        wk = TimeX.apply(w.cuda(), k.cuda(), B, C, T, 0)[...,1:].to(x.device)


        self.xx = xx
        self.bb = (wk[...,-1] - w[...,0,-1] * k[...,-1]) * extra_decay + k[...,-1]
        self.aa = (wkv[...,-1] - w[...,0,-1] * kv[...,-1]) * extra_decay + kv[...,-1]
        self.mm = mm
        assert not xx.isnan().any()
        assert not self.bb.isnan().any()
        assert not self.aa.isnan().any()
        assert not mm.isnan().any()

        rwkv = sigmoid(r) * torch.nan_to_num(wkv / wk).transpose(-1, -2)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        with torch.no_grad():  # init to "shift half of the channels"
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                x[0, 0, i] = 0
        self.time_mix = nn.Parameter(x)

        hidden_sz = 4 * config.n_embd
        self.key = Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = Linear(config.n_embd, config.n_embd, bias=False)
        self.value = Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def time_shift(self, x):
        return torch.cat([self.xx.expand(x.shape[0],x.shape[-1])[...,None,:], x], dim=-2)[...,:-1,:]

    def forward(self, x):
        xx = x[...,-1,:]
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)
        self.xx = xx

        k = self.key(x)
        k = torch.square(relu(k))
        kv = self.value(k)

        rkv = sigmoid(self.receptance(x)) * kv
        return rkv

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, layer_id+1000)
        else:
            self.att = RWKV_TimeMix(config, layer_id)

        self.ffn = RWKV_ChannelMix(config, layer_id)

    def forward(self, x):
        x = self.ln1(x)
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(x)  # better in some cases
        else:
            x = x + self.att(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = Linear(config.n_embd, config.vocab_size, bias=False)

        # self.head_q = Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        # self.head_q.scale_init = 0
        # self.head_k = Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        # self.head_k.scale_init = 0.1
        # self.register_buffer("copy_mask", torch.tril(
        #     torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        # RWKV_Init(self, config)

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))
        self.clear()

    def get_ctx_len(self):
        return self.ctx_len

    def clear(self, *batch_idcs):
        zeros = torch.zeros(1, self.config.n_embd, device=self.emb.weight.device)
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
    def save(self, *targets, device=None, detach=False):
        for target in targets:
            target.xx = {}
            target.aa = {}
            target.bb = {}
            target.mm = {}
        for idx, block in enumerate(self.blocks):
            for batch, target in enumerate(targets):
                for layer, valuename in [('ffn', 'xx'), ('att','xx'), ('att','aa'), ('att','bb'), ('att','mm')]:
                    value = getattr(getattr(block,layer), valuename)[batch]
                    if detach:
                        value = value.detach()
                    if device is not None:
                        value = value.to(device)
                    getattr(target, valuename)[f'{layer}.{idx}'] = value
    def load(self, *targets):
        device = self.emb.weight.device
        zeros = torch.zeros(self.config.n_embd, device=device)
        for idx, block in enumerate(self.blocks):
            block.ffn.xx = torch.stack([target.xx.get(f'ffn.{idx}', zeros).to(device) for target in targets])
            block.att.xx = torch.stack([target.xx.get(f'att.{idx}', zeros).to(device) for target in targets])
            block.att.aa = torch.stack([target.aa.get(f'att.{idx}', zeros).to(device) for target in targets])
            block.att.bb = torch.stack([target.bb.get(f'att.{idx}', zeros).to(device) for target in targets])
            block.att.mm = torch.stack([target.mm.get(f'att.{idx}', zeros).to(device) if hasattr(target, 'mm') else zeros for target in targets])

    def _init_weights(self, module):
        if isinstance(module, (Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None, recur=False):
        #import pdb; pdb.set_trace()
        if not recur:
            self.clear()
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)
        # q = self.head_q(x)[:, :T, :]
        # k = self.head_k(x)[:, :T, :]
        # c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)
        # c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

        # c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).float()
        # x = self.head(x) + c

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return x, loss
