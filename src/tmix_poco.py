import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .CoreDependencies import *
from .cuda6 import RUN_CUDA_RWKV6

from .tmix import TimeMixState

import math

from typing import Tuple

def causal_bias_mask(T):
    return torch.full((T, T), float('-inf')).triu(1)

def alibi_mask(T, H):
    bias = (torch.arange(T)[None, :] - torch.arange(T)[:, None]).float() # (T, T)
    bias = bias + causal_bias_mask(T) # (T, T)
    bias = bias.expand(H, -1, -1) # (H, T, T)
    head_bias_slopes = (2 ** torch.linspace(-8.0/H, -8.0, H)).unsqueeze(-1).unsqueeze(-1) # (H, 1, 1)
    bias = bias * head_bias_slopes # (H, T, T)
    return bias

class AlibiMask(nn.Module):
    def __init__(self, block_size : int, n_heads : int, layer_id : int):
        super().__init__()
        T = block_size
        H = n_heads
        self.register_buffer('mask', alibi_mask(T, H))

    def forward(self, q:Tensor):
        return self.mask[:, :q.size(-2), :q.size(-2)]

class RMSNorm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.dim = dim
        starting_scale = dim ** -0.5
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim) * starting_scale))
        else:
            self.scale = starting_scale

    def forward(self, x):
        assert(self.dim == x.size(-1))
        rms_norm = self.scale * x.norm(2, dim=-1, keepdim=True)
        return x / rms_norm.clamp(self.eps)
    
def rms_norm(x, eps:float = 1e-8):
    rms_norm = (x.size(-1) ** -0.5) * x.norm(2, dim=-1, keepdim=True)
    return x / (rms_norm + eps)

class Norm(nn.Module):
    def __init__(self, dim : int, weight_scaling : bool = True, eps = 1e-8):
        super().__init__()
        self.eps = eps
        if weight_scaling:
            self.register_parameter("scale", nn.Parameter(torch.ones(dim)))
        else:
            self.scale = 1

    def forward(self, x):
        return self.scale * x / x.norm(2, dim=-1, keepdim=True).clamp(self.eps)

def l2_norm(x, eps:float = 1e-8):
    # assumes that vector 'normally' has length 1, not length vec.size(-1)**0.5 (which would be if every component had an average absolute value of 1!)
    return x / (x.norm(2, dim=-1, keepdim=True) + eps)

class RWKV_Tmix_poco(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_layer = args.n_layer

        self.k_head_size = self.v_head_size = self.head_size = args.head_size_a
        self.n_kv_head = self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            self.time_maa_v_cache = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_q_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA))
            self.time_maa_q_w2 = nn.Parameter(torch.empty(D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_kv_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*2))
            self.time_maa_kv_w2 = nn.Parameter(torch.empty(2, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_r = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        #self.bias_mask = AlibiMask(args.ctx_len, self.n_kv_head, layer_id)

    @MyFunction
    def forward(self, x, kv_cache, last_time_mix_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        mq = torch.tanh(xxx @ self.time_maa_q_w1) @ self.time_maa_q_w2

        k, v = kv_cache.chunk(2, dim=-1)
        dv_prev = self.time_shift(v) - v
        xxx = v + dv_prev * self.time_maa_v_cache
        xxx = torch.tanh(xxx @ self.time_maa_kv_w1).view(B*T, self.time_maa_kv_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_kv_w2).view(self.time_maa_kv_w2.size(0), B, T, C)
        mk, mv = xxx.unbind(dim=0)

        dkprev = self.time_shift(k) - k
        dvprev = self.time_shift(v) - v

        xq = x + dxprev * (self.time_maa_r + mq)
        k = k + dkprev * (self.time_maa_k + mk)
        v = v + dvprev * (self.time_maa_v + mv)
        
        q = self.receptance(xq)
        
        q = self.ln_r(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(B,-1,H,K).transpose(1,2)
        k = k.view(B,-1,H,K).transpose(1,2)
        v = v.view(B,-1,H,V).transpose(1,2)

        # causality MUST be enforced for longer runs because even though we won't use the results at t-1 the next chanmix WILL for its tokenshift!
        # this is also why we must allow through the last MANY time-steps if we have that many, so chanmix receives both of these and can lerp between those results!
        # the results can tokenshift their way forward up to one full timestep each layer via chanmix, so we really have to keep up to all N poco layers around

        x = nn.functional.scaled_dot_product_attention(q,k,v,is_causal=q.size(-2)>1)

        x = x.transpose(1,2).reshape(B,-1,C)
       
        x = self.ln_x(x)
        #x = F.layer_norm(x.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(x.dtype)

        x = self.output(x)

        return x, TimeMixState(last_time_mix_state.wkv_state, shift_state)
