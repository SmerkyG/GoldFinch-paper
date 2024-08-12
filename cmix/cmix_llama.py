import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

from src.state import ModelState, ChannelMixState

class CMix_llama(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.dim_ffn = args.n_embd * 4 * 2 // 3 // 32 * 32

        self.w1 = nn.Linear(args.n_embd, self.dim_ffn, bias=False)
        self.w2 = nn.Linear(self.dim_ffn, args.n_embd, bias=False)
        self.w3 = nn.Linear(args.n_embd, self.dim_ffn, bias=False)

    def forward(self, x, last_model_state:ModelState):
        last_state = last_model_state.block_states[self.layer_id].channel_mix_state
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), last_state
