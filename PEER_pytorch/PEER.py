from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import einsum
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# rmsnorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main class

class PEER(Module):
    """
    following Algorithm 1 in the paper
    """

    def __init__(
        self,
        dim,
        *,
        heads = 8,                       # tested up to 32 - (hk = heads * num_experts_per_head (16))
        num_experts = 1_000_000,         # he chose 1 million
        num_experts_per_head = 16,       # he settled on 16, but was 32 in PKM paper
        activation = nn.GELU,
        dim_key = None,
        product_key_topk = None,
        separate_embed_per_head = False, # @smerky notes that heads may retrieve same redundant neurons. this setting would allow for separate embeds per head and prevent that
        pre_rmsnorm = False,
        dropout = 0.
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """

        super().__init__()

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # whether to do separate embedding per head

        num_expert_sets = 1 if not separate_embed_per_head else heads

        self.heads = heads
        self.separate_embed_per_head = separate_embed_per_head
        self.num_experts = num_experts

        # experts that will form the mlp project in / out weights

        self.weight_down_embed = nn.Embedding(num_experts * num_expert_sets, dim)
        self.weight_up_embed = nn.Embedding(num_experts * num_expert_sets, dim)

        # activation function, defaults to gelu

        self.activation = activation()

        # queries and keys for product-key

        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(sqrt(num_experts))

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias = False),
            Rearrange('b n (p h d) -> p b n h d', p = 2, h = heads)
        )

        self.product_key_topk = min(default(product_key_topk, num_experts_per_head), self.num_keys)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        # dropout

        #self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        # queries

        queries = self.to_queries(x)

        # first get similarity with keys

        sim = torch.einsum('p b n h d, h k p d -> p b n h k', queries, self.keys)

        # product key logic

        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim = -1)

        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head, dim = -1)

        indices = all_indices.gather(-1, pk_indices) # (b n h k) where k is num top-k experts chosen per head

        # if separate embeds per head, add appropriate offsets per head

        if self.separate_embed_per_head:
            head_expert_offsets = torch.arange(self.heads, device = x.device) * self.num_experts
            indices = einx.add('b n h k, h -> b n h k', indices, head_expert_offsets)

        # build the weight matrices for projecting in and out
        # basically the experts are the gathered parameters for an MLP

        weights_down = self.weight_down_embed(indices)
        weights_up = self.weight_up_embed(indices)

        # below is basically Algorithm 1 in paper

        x = torch.einsum('b n d, b n h k d -> b n h k', x, weights_down)

        x = self.activation(x)
        #x = self.dropout(x)

        x = x * scores.softmax(dim = -1)

        x = torch.einsum('b n h k, b n h k d -> b n d', x, weights_up)

        return x
