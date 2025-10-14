# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RWKV7Qwen3 model."""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, CacheLayerMixin
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs

from .configuration_rwkv7qwen3 import RWKV7Qwen3Config

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm, Qwen3Attention

class RWKV7State():
    def __init__(self) -> None:
        #super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.layer_kv_states: List[torch.Tensor] = []
        self.layer_shift_states:  List[torch.Tensor] = []
        self.cumulative_scores: List[torch.Tensor] = []
        self.sin: List[torch.Tensor] = []
        self.cos: List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.layer_kv_states)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Linear Attention variants do not have a maximum length
        return new_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError('Cannot reorder Linear Attention state')

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    def crop(self, max_length: int):
        # can't implement this for linear attention variants
        return

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the mask"""
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset
    
    @property
    def is_compileable(self) -> bool:
        """Return whether the cache is compileable"""
        return True #all(layer.is_compileable for layer in self.layers)
        
    @torch.no_grad
    def update(
        self,
        kv_state: torch.Tensor,
        shift_state: torch.Tensor,
        layer_idx: int,
        token_count: int = 0,
        is_attention_layer: bool = True,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:        
        # Update the number of seen tokens
        if layer_idx == 0:
            if is_attention_layer:
                token_count = kv_state.size(-2)
            self._seen_tokens += token_count

        # Update the cache
        if kv_state is not None:
            # There may be skipped layers, fill them with empty lists
            if layer_idx >= len(self.layer_kv_states):
                for _ in range(len(self.layer_kv_states), layer_idx):
                    if is_attention_layer:
                        self.layer_kv_states.append(torch.tensor([], dtype=kv_state.dtype, device=kv_state.device)) # acts as key_cache
                        self.layer_shift_states.append(torch.tensor([], dtype=shift_state.dtype, device=shift_state.device)) # acts as value_cache
                    else:
                        self.layer_kv_states.append(torch.zeros_like(kv_state).requires_grad_(False))
                        self.layer_shift_states.append(torch.zeros_like(shift_state).requires_grad_(False))
                self.layer_kv_states.append(kv_state) # acts as key_cache
                self.layer_shift_states.append(shift_state) # acts as value_cache
            else:
                if is_attention_layer:
                    self.layer_kv_states[layer_idx] = torch.cat([self.layer_kv_states[layer_idx], kv_state], dim=-2) # acts as key_cache
                    self.layer_shift_states[layer_idx] = torch.cat([self.layer_shift_states[layer_idx], shift_state], dim=-2) # acts as value_cache
                else:
                    self.layer_kv_states[layer_idx].copy_(kv_state)
                    self.layer_shift_states[layer_idx].copy_(shift_state)

        return self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx]

try:
    from fla.ops.rwkv7.chunk import chunk_rwkv7
    from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
except ImportError:
    print("Required module is not installed. Please install it using the following commands:")
    print("pip install --no-use-pep517 flash-linear-attention")
    print("Additionally, ensure you have at least version 2.2.0 of Triton installed:")
    print("pip install triton>=2.2.0")

def is_layer_attention(config, layer_id):
    return layer_id >= config.first_attention_layer and layer_id < config.first_post_attention_layer and  (layer_id > min(config.num_hidden_layers, config.last_striping_layer) or (min(config.num_hidden_layers-1, config.first_post_attention_layer, config.last_striping_layer) - layer_id) % config.attention_striping == 0)

class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: RWKV7Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rms_norm(hidden_states, eps = 1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_single(x, cos, sin, unsqueeze_dim=1):
    return (x * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(x) * sin.unsqueeze(unsqueeze_dim))

from typing import Callable, Optional, Tuple, Union
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask
from functools import lru_cache

block_mask = None

compiled_flex_attention = None
def get_flex_attention():
    global compiled_flex_attention
    if compiled_flex_attention == None:
        compiled_flex_attention = torch.compile(flex_attention)
    return compiled_flex_attention

SLIDING_WINDOW = 1024
SINK_WINDOW = 256

offset = torch.tensor(0).cuda()
def get_mask_mod_w_offset(mask_mod, _offset):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)
    return _mask_mod

def get_sliding_window_mask(L, S, device):
    # FIXME - stupid training mask generated by create_mask appears to be off by one e.g. 0 sized window would still have the current token in the window
    actual_window_size = SLIDING_WINDOW + 1
    assert S==L or L==1, "bad q length versus kv length"
    if S == L:
        ones = torch.ones([L,S], dtype=torch.bool, device=device)
        mask = ones.tril() ^ ones.tril(diagonal=-actual_window_size)
    else:
        mask = torch.ones([L,S], dtype=torch.bool, device=device)
        # FIXME - stupid training mask generated by create_mask appears to be off by one e.g. 0 sized window would still have the current token in the window
        mask[:,:-actual_window_size] = False
    return mask

def get_causal_mask(L, S, device):
    ones = torch.ones([S,S], dtype=torch.bool, device=device)
    causal = ones.tril()
    mask = causal
    return mask[-L:]

def get_swa_sink_mask(B, L, S, attention_mask, sliding_window, device):
    q_idx = torch.arange(S-L, S, device=device)[None, None, :, None]
    kv_idx = torch.arange(S, device=device)[None, None, None, :]
    window_mask = kv_idx >= q_idx - sliding_window
    if attention_mask is not None:
        assert attention_mask.dtype == torch.bool
        sink_indices = (S - attention_mask.view(B,L,S)[:,-1,:].sum(dim=-1)).view(B) # sink offset per batch idx
        sink_mask = kv_idx == sink_indices.view(B,1,1,1)
    else:
        sink_mask = kv_idx == 0
        attention_mask = kv_idx <= q_idx # causal
    return attention_mask & (sink_mask | window_mask)

# def get_swa_sink_mask(L, S, device):
#     ones = torch.ones([S,S], dtype=torch.bool, device=device)
#     swa = ~ones.tril(diagonal=-SLIDING_WINDOW)
#     sink = ones.clone()
#     sink[:, SINK_WINDOW:] = False
#     causal = ones.tril()
#     mask = causal & (swa | sink)
#     return mask[-L:]

# def swa_sink_mask(b, h, q_idx, kv_idx):
#     causal_mask = q_idx >= kv_idx
#     window_mask = q_idx - kv_idx <= SLIDING_WINDOW
#     sink_mask = kv_idx < SINK_WINDOW
#     return causal_mask & (window_mask | sink_mask)

def sliding_window_causal(score, b, h, q_idx, kv_idx):
    return torch.where((q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW), score, -float("inf"))

def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

block_mask = None

def stable_softmax(x, dim):
    z = x - torch.clamp_min(x.max(dim=dim, keepdim=True)[0], 0) # WOW this clamp is actually needed or you can get problems vs sdpa implementations if there are no non -inf values
    numerator = z.exp()
    denominator = numerator.sum(dim=dim, keepdim=True) + 1e-8
    return numerator/denominator

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.float()
    #attn_weight = stable_softmax(attn_weight, dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(attn_weight.isnan(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!
    #attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value.float()

class Qwen3AttentionAdapted(Qwen3Attention):
    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     frozen_residual: torch.Tensor,
    #     v_first: Optional[torch.Tensor] = None, 
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     return super().forward(hidden_states, **kwargs)[0], v_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     sliding_window=self.sliding_window,  # diff with Llama
        #     **kwargs,
        # )

        #key_states = repeat_kv(key, module.num_key_value_groups)
        #value_states = repeat_kv(value, module.num_key_value_groups)

        if attention_mask is not None and attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : k.shape[-2]]

        # if attention_mask is not None:
        #     causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        #     attn_weights = attn_weights + causal_mask
        # mask = torch.ones([S,S], device=x.device, dtype=torch.bool).tril()
        # mask = mask[-L:,:S] # convert to [L,S] for non-square attention
        # mask = torch.zeros([L,S], device=x.device, dtype=v.dtype).masked_fill(~mask, float('-inf'))
        # if attention_mask is not None:
        #     if attention_mask.dtype == torch.bool:
        #         attention_mask = torch.zeros_like(attention_mask, dtype=v.dtype).masked_fill(~attention_mask, float('-inf'))
        #     mask = attention_mask + mask
        # if self.layer_idx == 0:
        #     print(attention_mask[2])
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.zeros_like(attention_mask, dtype=v.dtype).masked_fill(~attention_mask, float('-inf'))

        scaling = q.size(-1) ** -0.5
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = attn_weights + attention_mask
        # attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!

        # #attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        y = torch.matmul(attn_weights, v)
        # y = attn_output.transpose(1, 2).contiguous()

        #y, _ = eager_attention_forward(self, q, k, v, attention_mask, scaling)

        from torch.nn.attention import SDPBackend, sdpa_kernel
        #with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        # y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask, is_causal=False, scale=scaling)
        # y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal= L == S)
        #y = scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=False, scale=scaling).to(v.dtype)
        y = y.transpose(1,2)
        
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first


class Qwen3AttentionVerticalSparse(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        QH, N = q.size(1), q.size(3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # assert past_key_values is None, "caching is not supported in this model"
        # if past_key_values is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"cache_position": cache_position}
        #     k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        S = k.size(-2)

        #y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal= L == S)

        scale = q.size(-1) ** -0.5

        a = q @ k.mT * scale

        ones = torch.ones([S,S], device=x.device, dtype=torch.bool)
        mask = torch.zeros([S,S], device=x.device, dtype=x.dtype).masked_fill(ones.triu(1),float('-inf'))
        mask = mask[-L:,:S] # convert to [L,S] for non-square attention
        a = a + mask

        #a = torch.softmax(a, dim=-1)
        #s = torch.softmax(a, dim=-1)

        #s = a.tril()

        s = a
        a = a.exp().to(v.dtype)

        #a = a.tril()
        #s = a

        n_first_tokens = 8
        attn_sum_required = 0.1

        # swa scores with window size n_first_tokens-1
        mask = torch.zeros([S,S], device=x.device, dtype=x.dtype).masked_fill(ones.triu(1) | ones.tril(1-n_first_tokens),float('-inf'))
        mask[:,0] = True # never mask sink token
        mask = mask[-L:,:S] # convert to [L,S] for non-square attention
        s = s + mask
        s = torch.softmax(s, dim=-1)
        #print(s[0,0])

        if past_key_values is not None:
            while len(past_key_values.cumulative_scores) <= self.layer_idx:
                past_key_values.cumulative_scores.append(torch.zeros(B, QH, n_first_tokens, device=q.device, dtype=q.dtype))
            cumulative_scores = past_key_values.cumulative_scores[self.layer_idx]
            key_cache = past_key_values.layer_kv_states[self.layer_idx]
            value_cache = past_key_values.layer_shift_states[self.layer_idx]

        scores_to_accumulate = s
        if S == L:
            # prefill, so test all columns up to col -n_first_tokens and delete those kv's as needed
            scores_to_accumulate = scores_to_accumulate.triu(-(n_first_tokens-1)) # mask off first 8 iterations (rows) for each seqpos (col)
            scores_to_accumulate = scores_to_accumulate.sum(-2, keepdim=True) 
            mask = scores_to_accumulate > attn_sum_required # for each column, check if any of the 8 iterations (first rows) are greater than zero dotproduct score
            #print('mask.mean().item()', mask.float().mean().item())
            #print(mask[0, 0, 0])
            # never mask n_first_tokens-1 rows of a given column because they haven't yet had time to accumulate enough history to know if they should be masked
            mask = mask | torch.ones_like(a).triu(-(n_first_tokens-1)).bool()
            # # FIXME - avoiding masking the sink token seems unnecessary, presumably because we measure it properly anyway so it doesnt get masked
            # never mask sink token
            #mask[:, :, :, 0] = True
            # FIXME - wrong way
            #mask[:, :, 0] = True
            a = a * mask
            #print(mask[0, 0])

        elif L == 1:
            # prefill, so test all columns up to col -n_first_tokens and delete those kv's as needed
            scores_to_accumulate = s
            pass
        else:
            # FIXME - handle L > 1 & S != L case
            assert False
          
        # if past_key_values is not None:
        #     scores_to_accumulate = scores_to_accumulate[:, :, -1, -n_first_tokens:] # cut down to last row and last n_first_tokens cols
        #     cumulative_scores += scores_to_accumulate

        #     if S == L:
        #         # apply last row of mask to kv cache entries
        #         kv_mask = mask[:, :, -1, :].unsqueeze(-1) # cut down to last row
        #         key_cache.masked_fill_(~kv_mask, float('-inf'))
        #         #value_cache *= kv_mask
        #     elif L == 1:
        #         pass
        #         # # permanently mask off the oldest iter out of n_first_tokens' kv cache entry if we haven't reached the required total attention score for it
        #         # kv_mask_oldest_iter = (cumulative_scores[:, :, 0] > attn_sum_required).unsqueeze(-1)
        #         # key_cache[:, :, -n_first_tokens] *= kv_mask_oldest_iter
        #         # value_cache[:, :, -n_first_tokens] *= kv_mask_oldest_iter
        #     else:
        #         # FIXME - handle L > 1 & S != L case
        #         assert False

        #     # roll cumulative scores left one slot
        #     past_key_values.cumulative_scores[self.layer_idx] = F.pad(cumulative_scores, [-1, 1])

        a = a / (a.sum(-1,keepdim=True) + 1e-8)
        y = a @ v

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

class Qwen3AttentionNoPE(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask, is_causal=attention_mask is None and L==S)
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

class Qwen3SymPow(Qwen3Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        n_embd = C = self.hidden_size = config.hidden_size
        H = self.num_heads = config.num_attention_heads
        N = self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        dim_att = H*N

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32

        if config.gate_rank_type == 1:
            self.gate = nn.Linear(self.hidden_size, dim_att, bias=False)
        elif config.gate_rank_type == 2:
            lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)
            self.g1 = nn.Parameter(torch.empty(n_embd, lora_rank_gate))
            self.g2 = nn.Parameter(torch.empty(lora_rank_gate, dim_att))

        if config.groupnorm_att:
            self.ln_x = nn.GroupNorm(self.num_heads, dim_att, eps=self.head_dim * 1e-5)

        if self.config.use_tokenshift:
            self.time_maa_x = nn.Parameter(torch.empty(1, 1, n_embd))
            self.time_maa_r = nn.Parameter(torch.empty(1, 1, n_embd))
            self.time_maa_k = nn.Parameter(torch.empty(1, 1, n_embd))
            self.time_maa_v = nn.Parameter(torch.empty(1, 1, n_embd))
            self.time_maa_w = nn.Parameter(torch.empty(1, 1, n_embd))
            self.time_maa_g = nn.Parameter(torch.empty(1, 1, n_embd))

            lora_rank_tokenshift = config.lora_rank_tokenshift or calc_lora_rank(0.5, 1.8)

            self.time_maa_w2 = nn.Parameter(torch.empty(5, lora_rank_tokenshift, n_embd))
            self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, lora_rank_tokenshift*self.time_maa_w2.size(0)))

        #lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        lora_rank_decay = config.lora_rank_decay or (64 if n_embd < 4096 else 128)

        # RWKV-6
        self.time_decay = nn.Parameter(torch.empty(H)) #dim_att))
        self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, lora_rank_decay))
        self.time_decay_w2 = nn.Parameter(torch.empty(lora_rank_decay, H)) #dim_att))  


    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RWKV7State] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        output_shift_state = hidden_states[:, -1:].detach().clone()
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)


        if self.config.use_tokenshift:
            if use_cache and past_key_values is not None and len(past_key_values) > self.layer_idx:
                input_kv_state, input_shift_state = past_key_values[self.layer_idx]
                xprev = torch.cat([input_shift_state[:, -1:], x[:, :-1]], dim=1)
            else:
                input_kv_state = None
                xprev = F.pad(x, (0, 0, 1, -1))
            dxprev = xprev - x

            xxx = x + dxprev * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*L, self.time_maa_w2.size(0), -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, L, -1)

            mr, mk, mv, mw, mg = xxx.unbind(dim=0)
            xq = x + dxprev * (self.time_maa_r + mr)
            xk = x + dxprev * (self.time_maa_k + mk)
            xv = x + dxprev * (self.time_maa_v + mv)
            xw = x + dxprev * (self.time_maa_w + mw)
            xg = x + dxprev * (self.time_maa_g + mg)
        else:
            xq = xk = xv = xw = xg = x

        q = self.q_norm(self.q_proj(xq).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(xk).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(xv).view(hidden_shape).transpose(1, 2)
        if self.config.gate_rank_type == 1:
            g = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        decay_states = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(q.dtype).unsqueeze(-1)
        log_w = -decay_states.float().exp()
        log_w = log_w.clamp(-5) # FIXME - is this necessary?
        #log_w = log_w.to(q.dtype)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # if past_key_values is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # if past_key_values is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     if position_embeddings is not None:
        #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     else:
        #         cache_kwargs = {"cache_position": cache_position}
        #     k_v_logw = torch.cat([k, v, log_w.view(B,L,self.num_key_value_heads,self.num_key_value_groups).transpose(1,2)], dim=-1)
        #     k_v_logw, output_shift_state = past_key_values.update(k_v_logw, output_shift_state, self.layer_idx, cache_kwargs)
        #     k, v, log_w = torch.split(k_v_logw, [k.size(-1), v.size(-1), self.num_key_value_groups], dim=-1)
        #     log_w = log_w.transpose(1,2).reshape(B,-1,self.num_heads,1)

        log_w = log_w.transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if position_embeddings is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            else:
                cache_kwargs = {"cache_position": cache_position}
            k_v_logw = torch.cat([k.float(), v.float(), log_w], dim=-1)
            k_v_logw, output_shift_state = past_key_values.update(k_v_logw, output_shift_state, self.layer_idx, cache_kwargs)
            k, v, log_w = torch.split(k_v_logw, [k.size(-1), v.size(-1), 1], dim=-1)
            k = k.to(q.dtype)
            v = v.to(q.dtype)

        S = k.size(-2)

        # log_w = log_w.view(B,self.num_heads,S,1)

        if self.config.balance_state:
            k = (k * (1 - log_w.exp())).to(k.dtype)

        #y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask) #is_causal= L == S)

        deg = 3
        norm = False

        scale = q.size(-1) ** -0.5

        #_qidx = torch.arange(S-L, S, device=q.device).unsqueeze(1)
        #_kidx = torch.arange(S, device=k.device).unsqueeze(0)
        m = attention_mask #_qidx >= _kidx
        s = (q @ k.mT) * scale
        signs = torch.sign(s)
        s = float(deg) * torch.where(m, torch.log(s.abs() + 1e-7), -float("inf"))
        if log_w is not None:
            log_w_sum = torch.cumsum(log_w.float(), dim=2)
            s = s + (log_w_sum[:, :, -L:, :] - log_w_sum.mT)
        rowmax = torch.max(s, dim=-1, keepdim=True).values.detach()
        rowmax = rowmax.masked_fill(rowmax.isinf(), 0) # IMPORTANT FOR BATCHED INFERENCE IN LM EVAL!        
        if deg % 2 == 0:
            p = torch.exp(s - rowmax).to(v.dtype)
        else:
            p = torch.exp(s - rowmax).to(v.dtype) * signs
        y = p @ v
        if norm:
            l = torch.sum(p, dim=-1)
            y = y / l[..., None]


        # #k, q, log_w = apply_phi(self.config.phi, k, q, log_w)

        # a = q @ k.mT * scale

        # a = a ** 2 # 1 + a + a ** 2 * 0.5 # squared for sym pow
        # #a = a.exp().to(v.dtype)

        # c = torch.cumsum(log_w.float(), dim=2)
        # c = (c[:, :, -L:, :] - c.mT)
        # if L == S:
        #     c = c.tril()
        # c = c.exp()
        # a = (a * c).to(v.dtype)

        # if L == S:
        #     a = a.tril()
        # a = a / (a.sum(-1,keepdim=True) + 1e-8)
        # y = a @ v

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()

        if self.config.groupnorm_att:
            #y = F.group_norm(y.view(B*L,-1).float(), num_groups=self.head_dim, weight=self.ln_x.weight.float(), bias=self.ln_x.bias.float(), eps = self.ln_x.eps).view(B,L,-1).to(v.dtype)
            y = self.ln_x(y.view(B * L, -1)).view(B, L, -1)
        #else:
            #y = y * scale
        if self.config.gate_rank_type != 0:
            y = y * g

        y = self.o_proj(y)

        attn_weights = None

        return y, v_first
    

        q = self.q_proj(xq)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        #decay_states = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(q.dtype)
        if self.config.gate_rank_type == 1:
            g = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            g = torch.sigmoid(xg @ self.g1) @ self.g2

        #decay_states = decay_states.view(B, L, self.num_heads) #self.head_dim)

        q = q.view(hidden_shape)
        k = k.view(hidden_shape)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is not None:
            q, k = q.transpose(1,2), k.transpose(1,2)
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q, k = q.transpose(1,2), k.transpose(1,2)

        q = q.view(*input_shape, -1)
        k = k.view(*input_shape, -1)

        # log_w = -decay_states.float().exp()
        # log_w = log_w.clamp(-5) # FIXME - is this necessary?

        # if past_key_values is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     if position_embeddings is not None:
        #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     else:
        #         cache_kwargs = {"cache_position": cache_position}
        #     k_v_logw = torch.cat([k, v, log_w], dim=-1)
        #     k_v_logw, output_shift_state = past_key_values.update(k_v_logw, output_shift_state, self.layer_idx, cache_kwargs)
        #     k, v, log_w = torch.split(k_v_logw, [k.size(-1), v.size(-1), log_w.size(-1)], dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        S = k.size(-2)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, S, self.num_key_value_heads, 1, -1).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, S, -1)
        v = v.view(B, S, self.num_key_value_heads, 1, -1).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, S, -1)

        # if self.config.balance_state:
        #     k = (k * (1 - log_w.exp())).to(k.dtype)

        scale = q.size(-1) ** -0.5

        #k, q, log_w = apply_phi(self.config.phi, k, q, log_w)

        q = q.view(hidden_shape)
        k = k.view(B,S,self.num_heads,-1)
        v = v.view(B,S,self.num_heads,-1)
        # log_w = log_w.view(B,S,self.num_heads,1)
        # q,k,v,log_w = map(lambda i: i.transpose(1,2).to(x.dtype), [q,k,v,log_w])
        q,k,v = map(lambda i: i.transpose(1,2).to(x.dtype), [q,k,v])

        a = q @ k.mT * scale

        #a = a ** 2 # 1 + a + a ** 2 * 0.5 # squared for sym pow

        # c = torch.cumsum(log_w, dim=2)
        # c = (c[:, :, -L:, :] - c.mT).tril().exp()
        # a = (a * c).to(v.dtype)

        a = a.exp().to(v.dtype)

        a = a.tril()
        a = a / (a.sum(-1,keepdim=True) + 1e-8)
        y = a @ v

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()

        if self.config.groupnorm_att:
            y = F.group_norm(y.view(B*L,-1).float(), num_groups=self.head_dim, weight=self.ln_x.weight.float(), bias=self.ln_x.bias.float(), eps = self.ln_x.eps).view(B,L,-1).to(v.dtype)
        #else:
            #y = y * scale
        if self.config.gate_rank_type != 0:
            y = y * g

        y = self.o_proj(y)

        return y, v_first

class Qwen3SWA(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        global SLIDING_WINDOW, SINK_WINDOW
        SLIDING_WINDOW = 1024
        SINK_WINDOW = 1

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=get_swa_sink_mask(B, L, S, attention_mask, SLIDING_WINDOW, q.device))
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

class Qwen3SWASink(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        
        # global block_mask
        # block_mask = create_block_mask(mask_mod=lambda b,h,q_idx,kv_idx: swa_sink_mask(b,h,q_idx,kv_idx), B=None, H=None, Q_LEN=L, KV_LEN=S, device=q.device)
        # y = get_flex_attention()(query=q, key=k, value=v, block_mask=block_mask)

        sliding_window = 1024

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=get_swa_sink_mask(B, L, S, attention_mask, sliding_window, q.device)) # attention_mask & get_swa_sink_mask(L, S, q.device))
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

class Qwen3DropoutSWASink(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)
        
        # global block_mask
        # block_mask = create_block_mask(mask_mod=lambda b,h,q_idx,kv_idx: swa_sink_mask(b,h,q_idx,kv_idx), B=None, H=None, Q_LEN=L, KV_LEN=S, device=q.device)
        # y = get_flex_attention()(query=q, key=k, value=v, block_mask=block_mask)

        global SLIDING_WINDOW, SINK_WINDOW
        SINK_WINDOW = 1
        SLIDING_WINDOW = 512
        swa_mask = get_swa_sink_mask(L, S, q.device)
        causal_mask = get_causal_mask(L, S, q.device)
        use_sliding = F.dropout(torch.ones(B, 1, L, 1, device=x.device), p=0.25) > 0
        chosen_mask = torch.where(use_sliding, swa_mask, causal_mask)
        # OH THIS BREAKS BECAUSE OF THE ATTENTION SINK ARGH
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & chosen_mask)
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first
    
class Qwen3MOBA(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)
        QH = self.config.num_attention_heads
        KVH = self.config.num_key_value_heads

        C = 64 # 512
        Z = (S+(C-1))//C*C

        n_topk = 1024 // C # max(1, int((Z//C)**0.5)) #16 #2

        n_topk = min(n_topk, Z // C)

        scale = q.size(-1) ** -0.5

        # do all mask ops on LxL for simplicity so we can use tril etc. and then later take the last S rows from it
        # chunked lower triangle -1 mask (without diagonal because we will always choose those and don't want them to pollute our top-k)
        mask = torch.ones([Z,Z], dtype=torch.bool, device=x.device).tril(-1)
        mask = mask.view(Z,Z//C,C)[:,:,0]
        mask = mask[S-L:S,:] # convert to [L,Z//C] for non-square masks

        # chunked softmax scores
        kc = F.pad(k, [0, 0, 0, Z-S])
        kc = kc.view(B, QH, Z//C, C, -1).mean(dim=-2) # B H Z//C headsize
        akc = q @ kc.mT * scale # B H L Q @ B H K Z//C = B H L Z//C
        # mask off everything outside the lower triangle, chunkwise, and also mask off the diagonal chunkwise so we don't score those due to using SWA on that part anyway
        akc = akc.masked_fill(~mask, float('-inf'))

        # # top-p
        # top_p = 0.125
        # akc = torch.softmax(akc, dim=-1)
        # akc = akc.masked_fill(akc.isnan(), 0)
        # sorted_logits, sorted_indices = torch.sort(akc, descending=True, dim=-1)
        # cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        # sorted_indices_to_remove = cumulative_probs > top_p
        # sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # sorted_indices_to_remove[..., 0] = 0
        # indices_to_remove = sorted_indices[sorted_indices_to_remove] # this is wrong, indexing doesnt work like this in pytorch for multidim tensors
        # print(indices_to_remove)
        # mask = torch.scatter(torch.ones_like(akc, dtype=torch.bool), -1, indices_to_remove, torch.zeros_like(akc, dtype=torch.bool))

        # top-k to mask
        values, indices = akc.topk(n_topk, dim=-1)
        mask = torch.scatter(torch.zeros_like(akc, dtype=torch.bool), -1, indices, torch.ones_like(values, dtype=torch.bool))
        #mask = akc > 0

        # if S == L:
        #     print("presence ratio ", mask.sum() / (mask.size(0) * mask.size(1) * mask.size(2) * mask.size(3) / 2) )
        #     print ("block mask", mask[0, 0, -1, :])

        # re-expand chunk mask to LxZ
        mask = mask.view(B, QH, L, Z//C, 1).expand(-1, -1, -1, -1, C).reshape(B, QH, L, Z)

        # always allow attention sink token
        mask[:, :, :, 0] = True

        ones = torch.ones([Z,Z], dtype=torch.bool, device=x.device)

        # always allow sliding window
        actual_window_size = C*2
        mask = mask | (ones.tril() ^ ones.tril(diagonal=-(actual_window_size+1)))[...,S-L:S,:]

        # convert to [L,S] for non-square attention
        mask = mask[...,:,:S]

        # causality
        #causal_mask = ones.tril()[S-L:S, :S]
        #mask = causal_mask
        #mask = mask & causal_mask

        # a = q @ k.mT * scale
        # a = a.masked_fill(~mask, float('-inf'))
        # a = torch.softmax(a, dim=-1)

        # y = a @ v

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & mask) #is_causal= L == S)

        #y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & get_sliding_window_mask(L, S, q.device))

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first
    
class Qwen3Power(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        C = 256
        window_size = 256
        window_size_in_chunks = window_size // C

        q_idx = torch.arange(S-L, S, device=x.device)[:, None]
        kv_idx = torch.arange(S, device=x.device)[None, :]
        mask_sink = kv_idx < C
        blk_qk = q_idx // C - kv_idx // C
        mask_window = blk_qk < window_size_in_chunks
        mask_power = (blk_qk & (blk_qk -1)) == 0
        mask_causal = q_idx >= kv_idx
        mask = mask_causal & (mask_window | mask_power | mask_sink)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & mask)

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first
    

class Qwen3Chunk(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        C = 256
        sink_size = 32
        window_size = 1025
        window_size_in_chunks = window_size // C

        q_idx = torch.arange(S-L, S, device=x.device)[:, None]
        kv_idx = torch.arange(S, device=x.device)[None, :]
        mask_sink = kv_idx < sink_size # C
        blk_qk = q_idx // C - kv_idx // C
        #mask_window = blk_qk < window_size_in_chunks
        mask_window = q_idx - kv_idx < window_size
        mask_causal = q_idx >= kv_idx
        mask_frontmost = q_idx == cache_position[-1]
        mask = mask_causal & mask_window
        #mask = mask_causal & (mask_sink | mask_window)
        #mask = mask_causal & (mask_sink | mask_window | mask_frontmost)
        #mask = mask_causal & (mask_window | mask_frontmost)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & get_sliding_window_mask(L, S, x.device))

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

class Qwen3SWAPrefill(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)

        sink_size = 1
        window_size = 256
        n_mha_tokens = 64

        #if S == L:
        q_idx = torch.arange(S-L, S, device=x.device)[:, None]
        kv_idx = torch.arange(S, device=x.device)[None, :]
        mask_sink = kv_idx < sink_size
        mask_window = q_idx - kv_idx < window_size
        mask_frontmost = q_idx >= S-n_mha_tokens # IMPORTANT or you would inference the first token using SWA not MHA!
        mask_causal = q_idx >= kv_idx
        mask = mask_causal & (mask_window | mask_sink | mask_frontmost)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & mask)
        #else:
        #    y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask)

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first

def int_quantize_and_unquantize(w: torch.Tensor, bitdepth:int) -> torch.Tensor:
    """(tp, x, y) -> (tp, y, x) (tp,)"""
    w_float = w.float()
    # Calculate max along time dim
    w_pos_max = w_float.amax(dim=[-2], keepdim=True).clamp(min=1e-12)
    w_neg_min = w_float.amin(dim=[-2], keepdim=True).clamp(max=-1e-12)
    # scale = 1 / abs_max    
    max_int = int(2**bitdepth)-1
    min_int = -int(2**bitdepth)
    pos_scale = w_pos_max / max_int
    neg_scale = -(w_neg_min / min_int)
    x_quant = (torch.relu(w_float) / pos_scale + torch.relu(-w_float) / neg_scale).round().to(torch.int)
    x_dequant = torch.relu(x_quant.float()) * pos_scale + torch.relu(-x_quant.float()) * neg_scale
    return x_dequant

def quantize_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(tp, x, y) -> (tp, y, x) (tp,)"""
    finfo = torch.finfo(torch.float8_e4m3fn)
    w_float = w.float()

    # Calculate max along time dim
    abs_max = (
        w_float.abs()
        .amax(dim=[-2], keepdim=True)
        .clamp(min=1e-12)
    )
    #abs_max = w_float.abs().clamp(min=1e-12)
    # scale = 1 / abs_max
    scale = torch.tensor([finfo.max], device=w.device) / abs_max

    # batch-wise scale and clamp
    # mul by inverse is faster than div
    x_scl_sat = (w_float * scale).clamp(min=finfo.min, max=finfo.max)
    x_scl_sat = x_scl_sat.to(torch.float8_e4m3fn) #.mT

    # un-invert scale
    scale = scale.float().reciprocal() #.squeeze()
    return (x_scl_sat, scale)

def quantize_fp8_maxima(w: torch.Tensor, abs_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(tp, x, y) -> (tp, y, x) (tp,)"""
    finfo = torch.finfo(torch.float8_e4m3fn)
    w_float = w.float()

    # scale = 1 / abs_max
    scale = torch.tensor([finfo.max], device=w.device) / abs_max

    # batch-wise scale and clamp
    # mul by inverse is faster than div
    x_scl_sat = (w_float * scale).clamp(min=finfo.min, max=finfo.max)
    x_scl_sat = x_scl_sat.to(torch.float8_e4m3fn) #.mT

    # un-invert scale
    scale = scale.float().reciprocal() #.squeeze()
    return (x_scl_sat, scale)

class Qwen3KeyQuant(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states

        B, L, D = x.size()

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)
        QH = q.size(1)
        KVH = k.size(1)
        N = k.size(-1)

        # quant_rope = False

        # if quant_rope:
        #     # normal version with rope applied before quantization
        #     cos, sin = position_embeddings
        #     q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # # FIXME - quantizing keys only on prefill, maybe try it everywhere if that works well
        # if L > 1:
        #     k_fp8_quant, k_fp8_scale = quantize_fp8(k)
        #     k = (k_fp8_quant.to(k) * k_fp8_scale).to(k)

        # if quant_rope:
        #     if past_key_values is not None:
        #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #         k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        # else:
        #     cos, sin = position_embeddings

        #     # q = apply_rotary_pos_emb_single(q, cos, sin)

        #     if past_key_values is not None:
        #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #         k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        #         # update cos, sin
        #         #while len(past_key_values.cos) <= self.layer_idx:
        #         #    past_key_values.cos.append(torch.tensor([], dtype=cos.dtype, device=cos.device))
        #         #    past_key_values.sin.append(torch.tensor([], dtype=sin.dtype, device=sin.device))
        #         if len(past_key_values.cos) <= self.layer_idx:
        #             past_key_values.cos.append(cos)
        #             past_key_values.sin.append(sin)
        #         else:
        #             cos = past_key_values.cos[self.layer_idx] = torch.cat([past_key_values.cos[self.layer_idx], cos], dim=-2)
        #             sin = past_key_values.sin[self.layer_idx] = torch.cat([past_key_values.sin[self.layer_idx], sin], dim=-2)

        #     q = apply_rotary_pos_emb_single(q, cos[..., -L:, :], sin[..., -L:, :])
        #     k = apply_rotary_pos_emb_single(k, cos, sin)

        # original code, did better even with uniform scale?
        # FIXME - quantizing keys only on prefill, maybe try it everywhere if that works well
        #if L > 1:
        # k_fp8_quant, k_fp8_scale = quantize_fp8(k)
        # k = k_fp8_quant.float() * k_fp8_scale # this forces us to store cache in float format, tho its what you'd get anyway when dequanting
        # k = k.float()

        # v_fp8_quant, v_fp8_scale = quantize_fp8(v)
        # v = v_fp8_quant.float() * v_fp8_scale # this forces us to store cache in float format, tho its what you'd get anyway when dequanting
        # v = v.bfloat16()

        if past_key_values is not None:
            seq_len_seen = past_key_values.layer_kv_states[self.layer_idx].size(-2) if len(past_key_values.layer_kv_states) > self.layer_idx else 0
            S = seq_len_seen + L
        else:
            S = k.size(-2)

        cos, sin = position_embeddings

        C = 32

        # torturous way to calculate the sink_mask
        sink_indices = (S - attention_mask.view(B,L,S)[:,-1,:].sum(dim=-1)).view(B) # sink offset per batch idx
        sink_mask = torch.zeros(B,S,dtype=attention_mask.dtype,device=attention_mask.device)
        sink_mask[torch.arange(B),sink_indices] = True
        sink_mask = sink_mask.view(B,1,1,S).expand(B,1,L,S)
        sink_mask = sink_mask & attention_mask

        # k_temp = k # save sink tokens
        # Z = (L+C-1)//C*C
        # if L >= C: k = F.pad(k, [0, 0, 0, Z-L]).view(B, KVH, Z//C, C, N)
        # k = int_quantize_and_unquantize(k, 3)
        # if L >= C: k = k.view(B, KVH, Z, N)[:, :, :L, :]
        # k = torch.where(sink_mask, k_temp, k) # or k[sink_mask] = k_temp[sink_mask]
        
        # v_temp = v
        # if L >= C: v = F.pad(v, [0, 0, 0, Z-L]).view(B, KVH, Z//C, C, N)
        # v = int_quantize_and_unquantize(v, 8).bfloat16()
        # if L >= C: v = v.view(B, KVH, Z, N)[:, :, :L, :]
        # v = torch.where(sink_mask, v_temp, v) # or v[sink_mask] = v_temp[sink_mask]

        q = apply_rotary_pos_emb_single(q, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

            # update cos, sin
            # while len(past_key_values.cos) <= self.layer_idx:
            #     past_key_values.cos.append(torch.tensor([], device=cos.device))            
            #     past_key_values.sin.append(torch.tensor([], device=sin.device))
            if len(past_key_values.cos) <= self.layer_idx:
                past_key_values.cos.append(cos)
                past_key_values.sin.append(sin)
            else:
                cos = past_key_values.cos[self.layer_idx] = torch.cat([past_key_values.cos[self.layer_idx], cos], dim=-2)
                sin = past_key_values.sin[self.layer_idx] = torch.cat([past_key_values.sin[self.layer_idx], sin], dim=-2)

        k = apply_rotary_pos_emb_single(k, cos, sin)
        q, k = q.to(v), k.to(v)


        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # if L == 1: # NOTE: the calling code ensures that this includes the first predicted token
        #     # remove all values that are not in the top-k qk scores
        #     n_topk = 256
        #     a = q @ k.mT
        #     a.masked_fill_(torch.ones(S, S, dtype=torch.bool, device=a.device).triu(1)[S-L:,:], float("-inf"))
        #     topk_values, topk_indices = a.topk(n_topk, dim=-1)
        #     topk_mask = torch.scatter(torch.zeros_like(a, dtype=torch.bool), -1, topk_indices, torch.ones_like(topk_values, dtype=torch.bool))
        #     v = v * topk_mask.view(B, QH, S, 1)

        scale = q.size(-1) ** -0.5
        a = q.float() @ k.mT.float() * scale
        a.masked_fill_(~attention_mask, float("-inf")) # torch.ones(S, S, dtype=torch.bool, device=a.device).triu(1)[S-L:,:]

        if L == 1: # NOTE: the calling code ensures that this includes the first predicted token
            # remove all values that are not in the top-k qk scores
            n_topk = 16 #64 # 64 was better than baseline
            always_last_n = 256
            #a2 = a.clone()
            #a2[..., 0] = float('-inf') # FIXME - FUCK, same usual stupid problem where sink token isn't always at zero, it's at the first unmasked (non-padded) position
            #print(a.shape, sink_mask.shape)
            sink_mask = sink_mask.view(B,1,L,S).expand(B,QH,L,S)
            a[sink_mask] = float('+inf') # was helpful
            a[..., -always_last_n:] = float('+inf')
            #all_true = torch.ones(S, S, dtype=torch.bool, device=a.device)
            #last_n_mask = (all_true.tril() ^ all_true.tril(always_last_n))[S-L:,:].view(1,1,L,S)
            #a.masked_fill_(last_n_mask, float('+inf'))
            topk_values, topk_indices = a.topk(n_topk+always_last_n+1, dim=-1)
            #print(torch.sort(topk_indices[0,0]).values)
            # #print(torch.zeros_like(k[..., :n_topk+always_last_n+1, :]).shape, topk_indices.shape, k.shape)
            # #k = topk_k = torch.scatter(torch.zeros_like(k[..., :n_topk+always_last_n+1, :]), -2, topk_indices.mT, k)
            # #v = topk_v = torch.scatter(torch.zeros_like(v[..., :n_topk+always_last_n+1, :]), -2, topk_indices.mT, v)
            # k = k[topk_indices.mT]
            # v = v[topk_indices.mT]
            # attention_mask = None #attention_mask[:, :, :, :(n_topk+always_last_n+1)]
            # print(q.shape, k.shape, v.shape)
            topk_mask = torch.scatter(torch.zeros_like(a, dtype=torch.bool), -1, topk_indices, torch.ones_like(topk_values, dtype=torch.bool))
            # # topk_mask[..., 0] = True # FIXME - FUCK, same usual stupid problem where sink token isn't always at zero, it's at the first unmasked (non-padded) position
            # # topk_mask[..., -always_last_n:] = True
            # # k = torch.cat([k[...,:1,:],k[...,-always_last_n:,:],topk_k], dim=-2)
            # # v = torch.cat([v[...,:1,:],v[...,-always_last_n:,:],topk_v], dim=-2)
            # # # FIXME - FUCK, same usual stupid problem where sink token isn't always at zero, it's at the first unmasked (non-padded) position
            # # attn_mask = attn_mask[:, :, :, ]
            # # print(q.shape, k.shape, v.shape)
            ##a.masked_fill_(~topk_mask, float("-inf"))

            v = v * topk_mask.view(B, QH, S, 1) # this performed better than attention masking, maybe because it doesn't impact the softmax scores
            #attention_mask = attention_mask & topk_mask

        # a = torch.softmax(a, -1)
        # a.masked_fill_(a.isnan(), 0) # bugfix for torch softmax behavior
        # y = a.to(v) @ v

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask)

        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        attn_weights = None

        return y, v_first
    
class Qwen3DoubleAttention(Qwen3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states
        if self.layer_idx == self.config.first_attention_layer:
            x = torch.cat([x, x], dim=1)

        B, L, D = x.size()
        L = L // 2

        w, x = x.view(B, 2, L, D).unbind(1)

        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # sliding window attention

        q = self.q_norm(self.q_proj(w).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(w).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(w).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)
        #print("L,S",L,S)

        global SLIDING_WINDOW, SINK_WINDOW
        SLIDING_WINDOW = 1024
        SINK_WINDOW = 1

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask & get_swa_sink_mask(L, S, q.device))
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y_w = self.o_proj(y)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     sliding_window=self.sliding_window,  # diff with Llama
        #     **kwargs,
        # )

        # global attention

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attention_mask) #is_causal= L == S)
        y = y.transpose(1,2)
        y = y.reshape(*input_shape, -1)#.contiguous()
        y = self.o_proj(y)

        if self.layer_idx < self.config.num_hidden_layers - 1:
            y = torch.cat([y_w, y], dim=1)

        attn_weights = None

        return y, v_first

class Qwen3NewAttention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: RWKV7Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        ddd = torch.empty(1, 1, config.hidden_size)
        self.time_maa_x = nn.Parameter(torch.empty_like(ddd))
        self.time_maa_k = nn.Parameter(torch.empty_like(ddd))
        self.time_maa_v = nn.Parameter(torch.empty_like(ddd))

        calc_lora_rank = lambda exponent, multiplier: max(1, round(config.hidden_size ** exponent * multiplier / 32)) * 32

        lora_rank_tokenshift = config.lora_rank_tokenshift or calc_lora_rank(0.5, 1.8)
        #lora_rank_tokenshift = 32 if n_embd < 4096 else 64

        self.time_maa_w2 = nn.Parameter(torch.empty(2, lora_rank_tokenshift, config.hidden_size))
        self.time_maa_w1 = nn.Parameter(torch.empty(config.hidden_size, lora_rank_tokenshift*self.time_maa_w2.size(0)))

        D_VALUE_LORA = max(config.hidden_size // 16, 64)
        self.time_key_w1 = nn.Parameter(torch.zeros(config.hidden_size, D_VALUE_LORA))
        self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, config.hidden_size).uniform_(-0.01, 0.01))
        self.time_value_w1 = nn.Parameter(torch.zeros(config.hidden_size, D_VALUE_LORA))
        self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, config.hidden_size).uniform_(-0.01, 0.01))

        C = config.hidden_size
        N = self.head_dim = config.head_dim
        H = self.num_heads = config.num_attention_heads
        calc_lora_rank = lambda exponent, multiplier: max(1, round(config.hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)

        self.v0 = nn.Parameter(torch.empty(1,1,H*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))

        self.mla_rank = min(config.hidden_size, config.num_key_value_heads * self.head_dim // 2) #min(config.hidden_size, 2 * config.num_key_value_heads * self.head_dim) #self.hidden_size# // 2 #min(self.hidden_size, self.num_heads * self.head_dim) # config.num_key_value_heads * self.head_dim
        self.kv_down_proj = nn.Parameter(torch.empty(config.hidden_size, self.mla_rank))
        self.kv_up_proj = nn.Parameter(torch.empty(self.mla_rank * 2, config.num_key_value_heads * self.head_dim * 2))


    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # dfrozen_residualprev = torch.nn.functional.pad(frozen_residual, (0, 0, 1, -1)) - frozen_residual

        # xxx = frozen_residual + dfrozen_residualprev * self.time_maa_x
        # xxx = torch.tanh(xxx @ self.time_maa_w1).view(hidden_states.shape[0]*hidden_states.shape[1], self.time_maa_w2.size(0), -1).transpose(0, 1)
        # xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), *hidden_states.shape)

        # mk, mv = xxx.unbind(dim=0)
        # xk = frozen_residual + dfrozen_residualprev * (self.time_maa_k + mk)
        # xv = frozen_residual + dfrozen_residualprev * (self.time_maa_v + mv)
        # #xk = xv = frozen_residual

        x = hidden_states

        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)

        kv_c = (x @ self.kv_down_proj)#.tanh()
        if self.layer_idx == self.config.first_attention_layer:
            v_first = torch.zeros_like(kv_c)
        kv_c_combined = torch.cat([v_first, kv_c], dim=-1)
        kv = kv_c_combined @ self.kv_up_proj
        k, v = torch.chunk(kv, 2, -1)
        v_first = kv_c

        k = self.k_norm(k.view(hidden_shape)).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, v_first
    
class RWKV7Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        C = self.hidden_size = config.hidden_size
        H = self.num_heads = config.num_attention_heads
        N = self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        lora_rank_iclr = config.lora_rank_iclr or calc_lora_rank(0.5, 1.8)
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)
        lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)

        # self.x_r = nn.Parameter(torch.empty(1,1,C))
        # self.x_w = nn.Parameter(torch.empty(1,1,C))
        # self.x_k = nn.Parameter(torch.empty(1,1,C))
        # self.x_v = nn.Parameter(torch.empty(1,1,C))
        # self.x_a = nn.Parameter(torch.empty(1,1,C))
        # self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,H*N))
        self.w1 = nn.Parameter(torch.empty(C, lora_rank_decay))
        self.w2 = nn.Parameter(torch.empty(lora_rank_decay, H*N))

        self.a0 = nn.Parameter(torch.empty(1,1,H*N))
        self.a1 = nn.Parameter(torch.empty(C, lora_rank_iclr))
        self.a2 = nn.Parameter(torch.empty(lora_rank_iclr, H*N))

        self.use_k_first = config.use_k_first
        v_first_headsize = self.num_key_value_heads if config.v_first_pre_gqa else H
        if self.use_k_first:
            self.k0 = nn.Parameter(torch.empty(1,1,v_first_headsize*N))
            self.k1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
            self.k2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, v_first_headsize*N))

        #if layer_id > 0:
        self.v0 = nn.Parameter(torch.empty(1,1,v_first_headsize*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, v_first_headsize*N))

        if config.gate_rank_type == 1:
            self.gate = nn.Linear(C, H*N, bias=False)
        elif config.gate_rank_type == 2:
            self.g1 = nn.Parameter(torch.empty(C, lora_rank_gate))
            self.g2 = nn.Parameter(torch.empty(lora_rank_gate, H*N))

        self.k_k = nn.Parameter(torch.empty(1,1,H*N))
        self.k_a = nn.Parameter(torch.empty(1,1,H*N))
        self.r_k = nn.Parameter(torch.empty(H,N))

        if self.config.groupnorm_att:
            self.ln_x = nn.GroupNorm(H, C, eps=self.head_dim * 1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RWKV7State] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) in (2, 4)
            # assert len(attention_mask.shape) == 2, (
            #     "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
            #     "for padding purposes (0 indicating padding). "
            #     "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            # )
        
        output_shift_state = hidden_states[:, -1:].detach().clone()

        x = hidden_states

        B, T, C = hidden_states.shape
        H = self.num_heads
        N = self.head_dim
        q_len = T

        if use_cache and past_key_values is not None and len(past_key_values) > self.layer_idx:
            input_vk_state, input_shift_state = past_key_values[self.layer_idx]
        else:
            input_vk_state, input_shift_state = torch.zeros(B,H,N,N, dtype=torch.float32,device=x.device), torch.zeros_like(x[:, -1:])

        xr = xw = xk = xv = xa = xg = x

        r = self.q_norm(self.q_proj(xr).view(B,T,-1,N))
        w_lora_result = self.w0 + (torch.tanh(xw @ self.w1) @ self.w2).float()
        k = self.k_norm(self.k_proj(xk).view(B,T,-1,N))
        v = self.v_proj(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        if self.config.gate_rank_type == 1:
            g = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            r, k = apply_rotary_pos_emb(r, k, cos, sin, unsqueeze_dim=2)

        if self.v0.shape[-2] == self.num_key_value_heads:
            if v_first is None:
                if self.use_k_first:
                    v_first = torch.cat([k,v], dim=-1)
                else:
                    v_first = v
            else:
                if self.use_k_first:
                    v_first_k, v_first_v = torch.chunk(v_first, 2, dim=-1)
                    k = k + (v_first_k - k) * torch.sigmoid(self.k0 + (xv @ self.k1) @ self.k2)
                    v = v + (v_first_v - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
                else:
                    v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        v = v.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        kk = (k).view(B,T,H,-1).float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)

        if self.v0.shape[-2] == self.num_heads:
            if v_first is None:
                if self.use_k_first:
                    v_first = torch.cat([k,v], dim=-1)
                else:
                    v_first = v
            else:
                if self.use_k_first:
                    v_first_k, v_first_v = torch.chunk(v_first, 2, dim=-1)
                    k = k + (v_first_k - k) * torch.sigmoid(self.k0 + (xv @ self.k1) @ self.k2)
                    v = v + (v_first_v - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
                else:
                    v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # dealing with left-padding
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                v = v * attention_mask[:, -v.shape[-2]:, None]
            elif len(attention_mask.shape) == 4:
                v = v * attention_mask[:, -1, -1, -v.shape[-2]:].view(B, T, 1)
                #v = v * attention_mask[:, :, -1, -v.shape[-2]:, None]

        log_w = -math.exp(-0.5) * torch.sigmoid(w_lora_result.float())       
        w = log_w.exp()

        if self.config.balance_state:
            k = k * (1-w+a)

        r,log_w,k,v,kk,a = [i.view(B,T,self.num_heads,-1) for i in [r,log_w,k,v,kk,a]]
        if self.training:
            x, output_vk_state = chunk_rwkv7(r, log_w, k, v, -kk, kk*a, initial_state=input_vk_state, output_final_state=use_cache)
        else:
            # if T == 1:
            #     output_vk_state = input_vk_state
            #     for t in range(T):
            #         r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
            #         vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
            #         ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
            #         output_vk_state = output_vk_state * w_.view(B,H,1,N) + output_vk_state @ ab.float() + vk.float()
            #         x[:,t] = (output_vk_state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)
            #     # FIXME - support fast triton kernel for non-training pre-fill with state in and out
            # else:
            x, output_vk_state = fused_recurrent_rwkv7(r, log_w, k, v, -kk, kk*a, initial_state=input_vk_state, output_final_state=use_cache)

        if self.config.groupnorm_att:
            x = torch.nn.functional.group_norm(x.view(B*T,H*N).float(), num_groups=H, weight=self.ln_x.weight.float(), bias=self.ln_x.bias.float(), eps = self.ln_x.eps).view(B,T,H*N).to(v.dtype)
        else:
            x = (x.view(B,T,H*N) * N ** -0.5).to(v.dtype)

        if self.config.use_bonus:
            x = x + ((r.to(v.dtype).view(B,T,H,-1)*k.to(v.dtype).view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,-1)

        if self.config.gate_rank_type != 0:
            x = x * g
        x = self.o_proj(x)

        if past_key_values is not None:
            past_key_values.update(output_vk_state, output_shift_state, self.layer_idx, q_len, is_layer_attention(self.config, self.layer_idx))

        return x, v_first
    
class RWKV7Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: RWKV7Qwen3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if is_layer_attention(config, layer_idx):
            att_fn = Qwen3Attention #Qwen3KeyQuant #Qwen3SWAPrefill #Qwen3DropoutSWASink #Qwen3AttentionNoPE #Qwen3MOBA #Qwen3AttentionVerticalSparse # Qwen3DoubleAttention # Qwen3SymPow #Qwen3Chunk #Qwen3Power #Qwen3MOBA #Qwen3Attention # Qwen3NewAttention # Qwen3AttentionAdapted
        else:
            att_fn = RWKV7Attention
        
        self.self_attn = att_fn(config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, v_first = self.self_attn(
            hidden_states=hidden_states,
            frozen_residual=frozen_residual,
            v_first=v_first,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            #is_causal=True,
        )

        if hidden_states.size(1) != residual.size(1):
            #print("ADJUSTING SIZE OF RESIDUAL FROM ", residual.size(1), " TO ", hidden_states.size(1))
            if hidden_states.size(1) > residual.size(1):
                assert hidden_states.size(1) == residual.size(1) * 2
                residual = torch.cat([residual, residual], dim=1)
            else:
                residual = F.pad(residual, [0, 0, hidden_states.size(1) - residual.size(1), 0])

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, v_first,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
  

@auto_docstring
class RWKV7Qwen3PreTrainedModel(PreTrainedModel):
    config: RWKV7Qwen3Config
    config_class = RWKV7Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

@auto_docstring
class RWKV7Qwen3Model(RWKV7Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: RWKV7Qwen3Config
    """

    def __init__(self, config: RWKV7Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RWKV7Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        
        # Initialize weights and apply final processing
        self.post_init()

    #@check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, RWKV7State):
            past_key_values = RWKV7State()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.use_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        v_first = None
        frozen_residual = None
        
        for decoder_layer in self.layers:
            if not is_layer_attention(self.config, decoder_layer.layer_idx):
                frozen_residual = rms_norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            attention_mask = causal_mask_mapping[decoder_layer.attention_type]
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = None

            layer_outputs = decoder_layer(
                hidden_states,
                frozen_residual=frozen_residual,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                v_first=v_first,
            )

            hidden_states = layer_outputs[0]
            if v_first is None:
                v_first = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        #if return_legacy_cache:
        #    next_cache = next_cache.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class RWKV7Qwen3ForCausalLM(RWKV7Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RWKV7Qwen3ForCausalLM

        >>> model = RWKV7Qwen3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # # run the prefill only up to the last token, then run one more for the actual result
        # # we do this so that called code doesn't have to handle the dichotomy specially and can just check for L==1
        # for i in range(2):
        #     all_but_one = max(1, input_ids.size(-1)-1)
        #     iid = input_ids[..., i*all_but_one:(i+1)*all_but_one]
        #     if iid.size(-1) == 0: 
        #         continue
        #     pids = position_ids
        #     if pids is not None:
        #         pids = position_ids[..., i*all_but_one:(i+1)*all_but_one]
        #     cp = cache_position
        #     if cp is not None:
        #         cp = cache_position[..., i*all_but_one:(i+1)*all_but_one]
        #     rv = self.forward_inner(iid, attention_mask=attention_mask, position_ids=pids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, cache_position=cp, num_logits_to_keep=num_logits_to_keep, **loss_kwargs)
        #     past_key_values = rv.past_key_values
    #     return rv

    # def forward_inner(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     num_logits_to_keep: int = 0,
    #     **loss_kwargs,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **loss_kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@auto_docstring
class RWKV7Qwen3ForSequenceClassification(RWKV7Qwen3PreTrainedModel):
    pass

@auto_docstring
class RWKV7Qwen3ForTokenClassification(RWKV7Qwen3PreTrainedModel):
    pass

@auto_docstring
class RWKV7Qwen3ForQuestionAnswering(RWKV7Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`
