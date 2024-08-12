import torch
from torch import nn, Tensor

from src.state import ModelState, ChannelMixState, Shared
from .cmix_rwkv_base import get_default_state

class CMix_x060loradapt(nn.Module):
    def get_default_state_factory(self): return get_default_state

    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        LORA = 64
        self.key_w1 = nn.Parameter(torch.zeros(args.n_embd, LORA))
        self.key_w2 = nn.Parameter(torch.zeros(LORA, args.dim_ffn).uniform_(-0.01, 0.01))
        self.receptance_w1 = nn.Parameter(torch.zeros(args.n_embd, LORA))
        self.receptance_w2 = nn.Parameter(torch.zeros(LORA, args.n_embd).uniform_(-0.01, 0.01))
        self.value_w1 = nn.Parameter(torch.zeros(args.dim_ffn, LORA))
        self.value_w2 = nn.Parameter(torch.zeros(LORA, args.n_embd).uniform_(-0.01, 0.01))

    def forward(self, x, last_model_state:ModelState, shared_key, shared_value, shared_receptance):
        last_state = last_model_state.block_states[self.layer_id].channel_mix_state
        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = xk @ shared_key.mT + torch.tanh(xk @ self.key_w1) @ self.key_w2 #  xk @ (shared.key + self.key.weight).mT # self.key(xk)
        k = torch.relu(k) ** 2
        kv = k @ shared_value.mT + torch.tanh(k @ self.value_w1) @ self.value_w2 # k @ (shared.value + self.value.weight).mT #self.value(k)
        r = xr @ shared_receptance.mT + torch.tanh(xr @ self.receptance_w1) @ self.receptance_w2 #torch.sigmoid(xr @ (shared.receptance + self.receptance.weight).mT)
        return torch.sigmoid(r) * kv, ChannelMixState(shift_state=shift_state)
