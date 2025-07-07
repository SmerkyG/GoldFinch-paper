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

from transformers.cache_utils import Cache, StaticCache, SlidingWindowCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_rwkv7qwen3 import RWKV7Qwen3Config

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3MLP, Qwen3RMSNorm, Qwen3Attention
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "recursal/QRWKV7-7B-Instruct-Preview-v0.1"
_CONFIG_FOR_DOC = "RWKV7QwenConfig"

class RWKV7State(Cache):
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.layer_kv_states: List[torch.Tensor] = []
        self.layer_shift_states:  List[torch.Tensor] = []

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
                        self.layer_kv_states.append(torch.tensor([])) # acts as key_cache
                        self.layer_shift_states.append(torch.tensor([])) # acts as value_cache
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
    return layer_id >= config.num_hidden_layers - config.num_attention_layers and (layer_id > min(config.num_hidden_layers, config.last_striping_layer) or (min(config.num_hidden_layers-1, config.last_striping_layer) - layer_id) % config.attention_striping == 0)

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

def generate_rotary_embedding(max_seqlen:int, dim:int, theta:float = 10000.0, scale:float = 1):
    #inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float).to(device) / dim))

    angular_velocity = theta ** -(torch.arange(0, dim, 2, dtype=torch.float) / dim) / scale # frequencies from 1.0 ... 1/theta
    angles = torch.outer(torch.arange(max_seqlen), angular_velocity)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((angles, angles), dim=-1)
    return torch.stack([emb.cos(), emb.sin()], dim=0)
    #return torch.polar(torch.ones_like(angles), angles)

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
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask

SLIDING_WINDOW = 1024

def sliding_window_causal(score, b, h, q_idx, kv_idx):
    return torch.where((q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW), score, -float("inf"))

def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

block_mask = None

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
        past_key_value: Optional[Cache] = None,
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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

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

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal= L == S)
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
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        x = hidden_states
        if self.layer_idx == self.config.num_hidden_layers - self.config.num_attention_layers:
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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        S = k.size(-2)
        #print("L,S",L,S)

        #global block_mask
        #if block_mask is None:
        assert S==L or L==1, "bad q length versus kv length"
        if S == L:
            # FIXME - stupid training mask generated by create_mask appears to be off by one like 0 SWA would still have the current token in the window
            ones = torch.ones([L,S], dtype=torch.bool, device=q.device)
            block_mask = ones.tril() ^ ones.tril(diagonal=-SLIDING_WINDOW-1)
            #block_mask = create_mask(mod_fn=sliding_window_causal_mask, B=None, H=None, Q_LEN=S, KV_LEN=L, device=q.device)
        else:
            block_mask = torch.ones([L,S], dtype=torch.bool, device=q.device)
            # FIXME - stupid training mask generated by create_mask appears to be off by one like 0 SWA would still have the current token in the window
            block_mask[:,:-SLIDING_WINDOW-1] = False
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=block_mask)
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
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal= L == S)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        dfrozen_residualprev = torch.nn.functional.pad(frozen_residual, (0, 0, 1, -1)) - frozen_residual

        xxx = frozen_residual + dfrozen_residualprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(hidden_states.shape[0]*hidden_states.shape[1], self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), *hidden_states.shape)

        mk, mv = xxx.unbind(dim=0)
        xk = frozen_residual + dfrozen_residualprev * (self.time_maa_k + mk)
        xv = frozen_residual + dfrozen_residualprev * (self.time_maa_v + mv)
        #xk = xv = frozen_residual

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(xk).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(xv).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
            query_states,
            key_states,
            value_states,
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

        #if layer_idx > 0:
        self.v0 = nn.Parameter(torch.empty(1,1,H*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))

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
        past_key_value: Optional[RWKV7State] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )
        
        output_shift_state = hidden_states[:, -1:].detach().clone()

        x = hidden_states

        B, T, C = hidden_states.shape
        H = self.num_heads
        N = self.head_dim
        q_len = T

        if use_cache and past_key_value is not None and len(past_key_value) > self.layer_idx:
            input_vk_state, input_shift_state = past_key_value[self.layer_idx]
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

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        v = v.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        kk = (k).view(B,T,H,-1).float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)

        if self.layer_idx == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)        

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]

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
        if self.config.gate_rank_type != 0:
            x = x * g
        x = self.o_proj(x)

        if past_key_value is not None:
            past_key_value.update(output_vk_state, output_shift_state, self.layer_idx, q_len, is_layer_attention(self.config, self.layer_idx))

        return x, v_first
    
class RWKV7Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: RWKV7Qwen3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if is_layer_attention(config, layer_idx):
            self.self_attn = Qwen3AttentionAdapted(config=config, layer_idx=layer_idx) # Qwen3DoubleAttention(config=config, layer_idx=layer_idx) # Qwen3NewAttention(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = RWKV7Attention(config, layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frozen_residual: torch.Tensor,
        v_first: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
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
            past_key_value=past_key_value,
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

    

RWKV7QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RWKV7Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    RWKV7QWEN3_START_DOCSTRING,
)
class RWKV7Qwen3PreTrainedModel(PreTrainedModel):
    config_class = RWKV7Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


RWKV7QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

@add_start_docstrings(
    "The bare RWKV7Qwen3 Model outputting raw hidden-states without any specific head on top.",
    RWKV7QWEN3_START_DOCSTRING,
)
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
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        #if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # if self.config.num_attention_layers > 0:
        #     causal_mask = self._update_causal_mask(
        #         attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        #     )
        # else:
        causal_mask = None

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
        
        for decoder_layer in self.layers:
            if not is_layer_attention(self.config, decoder_layer.layer_idx):
                frozen_residual = rms_norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    frozen_residual,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    v_first,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    frozen_residual=frozen_residual,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    v_first=v_first,
                )

            hidden_states = layer_outputs[0]
            v_first = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        #if return_legacy_cache:
        #    next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
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

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(RWKV7QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
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

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@add_start_docstrings(
    """
    The RWKV7Qwen3 Model transformer with a sequence classification head on top (linear layer).

    [`RWKV7Qwen3ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    RWKV7QWEN3_START_DOCSTRING,
)
class RWKV7Qwen3ForSequenceClassification(RWKV7Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV7Qwen3Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN3_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The RWKV7Qwen3 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    RWKV7QWEN3_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->RWKV7Qwen3, LLAMA->RWKV7Qwen3
class RWKV7Qwen3ForTokenClassification(RWKV7Qwen3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV7Qwen3Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN3_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
The RWKV7Qwen3 Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    RWKV7QWEN3_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralForQuestionAnswering with Mistral->RWKV7Qwen3, MISTRAL->RWKV7Qwen3
class RWKV7Qwen3ForQuestionAnswering(RWKV7Qwen3PreTrainedModel):
    base_model_prefix = "model"

    # Copied from models.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->RWKV7Qwen3
    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Qwen3Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )