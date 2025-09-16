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
"""PyTorch RWKV6Qwen2 model."""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, StaticCache, DynamicCache
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
from .configuration_rwkv6qwen2 import RWKV6Qwen2Config

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2MLP, Qwen2RMSNorm, repeat_kv
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "RWKV/RWKV6Qwen2-7B"
_CONFIG_FOR_DOC = "RWKV6Qwen2Config"

class RWKV6State():
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
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:        
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += token_count

        # Update the cache
        # There may be skipped layers, fill them with empty lists
        for _ in range(len(self.layer_kv_states), layer_idx + 1):
            self.layer_kv_states.append(torch.zeros_like(kv_state).requires_grad_(False))
            self.layer_shift_states.append(torch.zeros_like(shift_state).requires_grad_(False))
        self.layer_kv_states[layer_idx].copy_(kv_state)
        self.layer_shift_states[layer_idx].copy_(shift_state)

        return self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx]

try:
    #from fla.ops.gla.chunk import chunk_gla
    from fla.ops.gla.fused_recurrent import fused_recurrent_gla
except ImportError:
    print("Required module is not installed. Please install it using the following commands:")
    print("pip install --no-use-pep517 flash-linear-attention")
    print("Additionally, ensure you have at least version 2.2.0 of Triton installed:")
    print("pip install triton>=2.2.0")

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: RWKV6Qwen2Config, device=None):
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

def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            #nn.init.orthogonal_(x, gain=gain * scale)
            x.copy_(nn.init.orthogonal_(torch.empty_like(x, dtype=torch.float32), gain=gain * scale))
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                #nn.init.orthogonal_(x[i], gain=gain * scale)
                x[i].copy_(nn.init.orthogonal_(torch.empty_like(x[i], dtype=torch.float32), gain=gain * scale))                            
        else:
            assert False
        return x

class RWKV6Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        n_layer = self.config.num_hidden_layers
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_idx

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=getattr(config, 'attention_output_bias', config.attention_bias))

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32

        if config.gate_rank_type == 1:
            self.gate = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        elif config.gate_rank_type == 2:
            lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)
            self.g1 = nn.Parameter(torch.empty(n_embd, lora_rank_gate))
            self.g2 = nn.Parameter(torch.empty(lora_rank_gate, n_embd))

        if config.groupnorm_att:
            self.ln_x = nn.GroupNorm(self.num_heads, dim_att, eps=self.head_dim * 1e-5)

        with torch.no_grad():
            if config.gate_rank_type == 1:
                self.gate.weight.zero_()                
            elif config.gate_rank_type == 2:
                self.g1.zero_()
                ortho_init(self.g2, 0.1)

            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            
            if self.config.use_tokenshift:
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd

                ddd = torch.zeros(1, 1, n_embd)
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_r = nn.Parameter(torch.zeros_like(ddd))
                self.time_maa_k = nn.Parameter(torch.zeros_like(ddd))
                self.time_maa_v = nn.Parameter(torch.zeros_like(ddd))
                self.time_maa_w = nn.Parameter(torch.zeros_like(ddd))
                self.time_maa_g = nn.Parameter(torch.zeros_like(ddd))

                lora_rank_tokenshift = config.lora_rank_tokenshift or (32 if n_embd < 4096 else 64)

                self.time_maa_w2 = nn.Parameter(torch.zeros(5, lora_rank_tokenshift, n_embd).uniform_(-0.01, 0.01))
                self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, lora_rank_tokenshift*self.time_maa_w2.size(0)))

            lora_rank_decay = config.lora_rank_decay or (64 if n_embd < 4096 else 128)

            # RWKV-6
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, lora_rank_decay))
            self.time_decay_w2 = nn.Parameter(torch.zeros(lora_rank_decay, dim_att).uniform_(-0.01, 0.01))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RWKV6State] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        output_shift_state = hidden_states[:, -1:].detach().clone()

        x = hidden_states

        B, T, C = hidden_states.shape
        H = self.num_heads
        N = self.head_dim

        if use_cache and past_key_values is not None and len(past_key_values) > self.layer_idx:
            input_kv_state, input_shift_state = past_key_values[self.layer_idx]
            xprev = torch.cat([input_shift_state, x[:, :-1]], dim=1)
        else:
            input_kv_state = None
            xprev = F.pad(x, (0, 0, 1, -1))

        if self.config.use_tokenshift:
            dxprev = xprev - x

            xxx = x + dxprev * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

            mr, mk, mv, mw, mg = xxx.unbind(dim=0)
            xr = x + dxprev * (self.time_maa_r + mr)
            xk = x + dxprev * (self.time_maa_k + mk)
            xv = x + dxprev * (self.time_maa_v + mv)
            xw = x + dxprev * (self.time_maa_w + mw)
            xg = x + dxprev * (self.time_maa_g + mg)
        else:
            xr = xk = xv = xw = xg = x

        r = self.q_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        w_lora_result = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(r.dtype)
        if self.config.gate_rank_type == 1:
            g = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            g = torch.sigmoid(xg @ self.g1) @ self.g2

        if position_embeddings is not None:
            r = r.view(B,T,-1,N)
            k = k.view(B,T,-1,N)
            cos, sin = position_embeddings
            r, k = apply_rotary_pos_emb(r, k, cos, sin, unsqueeze_dim=2)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        v = v.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        log_w = -w_lora_result.float().exp()
        log_w = log_w.clamp(-5)
        if self.config.balance_state:
            k = (k * (1 - log_w.exp())).to(k.dtype)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, None, -v.shape[-2]:, None]

        r = r.view(B,T,-1,N).to(v.dtype)
        k = k.view(B,T,-1,N).to(v.dtype)
        v = v.view(B,T,-1,N)
        log_w = log_w.view(B,T,-1,N)

        attn_weights = torch.empty(0, device=x.device)

        scale = r.shape[-1] ** -0.5
        output_final_state = not self.training and use_cache and past_key_values is not None
        attn_output, output_kv_state = fused_recurrent_gla(r, k, v, log_w, None, scale, input_kv_state, output_final_state)

        if output_final_state:
            past_key_values.update(output_kv_state, output_shift_state, T, self.layer_idx)

        attn_output = attn_output.view(B, T, -1)
        if self.config.groupnorm_att:
            attn_output = self.ln_x(attn_output.view(B * T, -1)).view(B, T, -1)
        if self.config.gate_rank_type != 0:
            attn_output = attn_output * g
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
    
class RWKV6Qwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: RWKV6Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = RWKV6Attention(config, layer_idx) #QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

RWKV6QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RWKV6Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    RWKV6QWEN2_START_DOCSTRING,
)
class RWKV6Qwen2PreTrainedModel(PreTrainedModel):
    config_class = RWKV6Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV6Qwen2DecoderLayer"]
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


RWKV6QWEN2_INPUTS_DOCSTRING = r"""
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
    "The bare RWKV6Qwen2 Model outputting raw hidden-states without any specific head on top.",
    RWKV6QWEN2_START_DOCSTRING,
)
class RWKV6Qwen2Model(RWKV6Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: RWKV6Qwen2Config
    """

    def __init__(self, config: RWKV6Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RWKV6Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV6QWEN2_INPUTS_DOCSTRING)
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        #return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, RWKV6State):
            #return_legacy_cache = True
            past_key_values = RWKV6State()
            # if past_key_values is None:
            #     past_key_values = DynamicCache()
            # else:
            #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            #     logger.warning_once(
            #         "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
            #         "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
            #         "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            #     )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        causal_mask = None

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = None
        if self.config.use_rope:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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

class RWKV6Qwen2ForCausalLM(RWKV6Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV6Qwen2Model(config)
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

    @add_start_docstrings_to_model_forward(RWKV6QWEN2_INPUTS_DOCSTRING)
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
        >>> from transformers import AutoTokenizer, RWKV6Qwen2ForCausalLM

        >>> model = RWKV6Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            'past_key_values': past_key_values,
            'attention_mask': attention_mask,
            'cache_position': cache_position,
        }
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs['inputs_embeds'] = inputs_embeds
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs['input_ids'] = input_ids.contiguous()

        model_inputs.update(**kwargs)

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)

        return model_inputs        

@add_start_docstrings(
    """
    The RWKV6Qwen2 Model transformer with a sequence classification head on top (linear layer).

    [`RWKV6Qwen2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    RWKV6QWEN2_START_DOCSTRING,
)
class RWKV6Qwen2ForSequenceClassification(RWKV6Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV6Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV6QWEN2_INPUTS_DOCSTRING)
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
    The RWKV6Qwen2 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    RWKV6QWEN2_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->RWKV6Qwen2, LLAMA->RWKV6QWEN2
class RWKV6Qwen2ForTokenClassification(RWKV6Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV6Qwen2Model(config)
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

    @add_start_docstrings_to_model_forward(RWKV6QWEN2_INPUTS_DOCSTRING)
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
The RWKV6Qwen2 Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    RWKV6QWEN2_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralForQuestionAnswering with Mistral->RWKV6Qwen2, MISTRAL->RWKV6QWEN2
class RWKV6Qwen2ForQuestionAnswering(RWKV6Qwen2PreTrainedModel):
    base_model_prefix = "model"

    # Copied from models.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->RWKV6Qwen2
    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV6Qwen2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV6QWEN2_INPUTS_DOCSTRING)
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