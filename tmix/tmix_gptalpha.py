import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.state import ModelState, TimeMixState, Shared

from src.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding

from configs import Transformer_Config
def get_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return TimeMixState(
        torch.zeros([2, B, 0, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad), 
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad),
    )

class TMix_gptalpha(nn.Module):
    def get_default_state_factory(self): return get_default_state
    
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.n_layer = args.n_layer

        self.k_head_size = self.v_head_size = self.head_size = args.head_size
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
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            D_MIX_LORA = 32 if args.n_embd < 4096 else 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*3))
            self.time_maa_w2 = nn.Parameter(torch.zeros(3, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            D_DECAY_LORA = 64 if args.n_embd < 4096 else 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att//2).uniform_(-0.01, 0.01))

        self.query = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.ln_r = nn.LayerNorm(args.dim_att)
        self.ln_k = nn.LayerNorm(args.dim_att)
        self.ln_v = nn.LayerNorm(args.dim_att)
        self.ln_x = nn.LayerNorm(args.dim_att)

    def forward(self, x, xo, kv_cache, last_model_state:ModelState, shared:Shared):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, T, C = x.shape
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x

        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        mr, mk, mv = xxx.unbind(dim=0)
        xq = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        
        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        
        q = self.ln_r(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        # handle recurrent inference via maintaining a kv cache
        wkv_state = last_state.wkv_state
        if not self.training:
            new_kv_cache = torch.stack([k, v], dim=0)
            wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
            k, v = wkv_state.unbind(0)
            k, v = k.contiguous(), v.contiguous()

        is_causal = q.size(1)==k.size(1)

        q = q.view(B,-1,H,K).transpose(1,2)
        k = k.view(B,-1,H,K).transpose(1,2)
        v = v.view(B,-1,H,V).transpose(1,2)

        q, k = apply_rotary_embedding(q, k, shared.angles)

        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        x = x.transpose(1,2).reshape(B,T,C)
       
        x = self.ln_x(x)

        x = self.output(x)

        return x, TimeMixState(wkv_state, shift_state)

