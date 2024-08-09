import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

from typing import Tuple

def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def generate_dynamic_rotary_embedding(positions:Tensor, dim:int, theta:float = 10000.0, scale:float = 1):
    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float, device=positions.device) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = positions * angular_velocity
    return torch.polar(torch.ones_like(angles), angles)

def generate_binary_rotary_embedding(max_seqlen:int, dim:int, scale:float=1):
    arange = torch.arange(dim // 2)
    angular_velocity = math.pi * (2.0 ** -arange) / scale # fastest velocity will rotate fully in two steps
    angular_velocity[int(math.log2(max_seqlen)):] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across the seqlen
    #angular_velocity[20:] = 0.0 # don't supply velocities slower than the one that will get a single full rotation across 1024k ctxlen
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    return torch.polar(torch.ones_like(angles), angles)

def apply_single_rotary_embedding(q, angles, seq_dim:int = -2) -> Tensor:
    if angles.size(0) == 0:
        return q
    
    q_dtype = q.dtype
    L = q.size(seq_dim)
    #q_angles = angles[-L:].view(1, 1, L, angles.size(1))
    q_angles = angles
    if q.ndim == 3:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), -1, 2)) * q_angles
    else:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2)) * q_angles

    return torch.view_as_real(q).flatten(-2).to(q_dtype)

def apply_rotary_embedding(q, k, angles, seq_dim:int = -2) -> Tuple[Tensor, Tensor]:
    if angles.size(0) == 0:
        return q, k
    
    q_dtype, k_dtype = q.dtype, k.dtype
    L = q.size(seq_dim)
    q_angles = angles[-L:].view(1, 1, L, angles.size(1))
    if q.ndim == 3:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), -1, 2)) * q_angles
    else:
        q = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2)) * q_angles

    L = k.size(seq_dim)
    k_angles = angles[-L:].view(1, 1, L, angles.size(1))
    if k.ndim == 3:
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), -1, 2)) * k_angles
    else:
        k = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2)) * k_angles

    return torch.view_as_real(q).flatten(3).to(q_dtype), torch.view_as_real(k).flatten(3).to(k_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, max_sequence_length:int, dim:int, seq_dim:int = -2, theta:float = 10000):
        super().__init__()
        self.angles = generate_rotary_embedding(max_sequence_length, dim, theta)
        self.seq_dim = seq_dim

    def forward(self, q, k):
        return apply_rotary_embedding(q, k, self.angles, self.seq_dim)
