import os, math, gc, importlib.util
import torch
import torch.utils.checkpoint
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, Optional

from src.state import ModelState, BlockState, ChannelMixState, TimeMixState, Shared

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

from src.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding

from src.CoreDependencies import *

from dataclasses import dataclass

import torch.utils.checkpoint
if importlib.util.find_spec('deepspeed'):
    import deepspeed

from src.logger import print0 as print

from fla.ops.simple_gla.chunk import chunk_simple_gla, SimpleGLAFunction

def fla_chunk_simple_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dim() == k.dim() == v.dim() == 4, "q, k, v must have 4 dimensions (b, h, l, d)"
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    scale = k.shape[-1] ** -0.5
    g = g.float()
    initial_state = None
    output_final_state = False
    o, final_state = SimpleGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    return o, final_state

from fla.ops.gla.chunk import chunk_gla, ChunkGLAFunction

def fla_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    g = g.float()
    initial_state = None
    output_final_state = False
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    return o, final_state

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

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


# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim:int=1):
    B, L = q.size(0), q.size(-2)
    cos = cos[:L].unsqueeze(0).expand(B,L,-1).unsqueeze(unsqueeze_dim)
    sin = sin[:L].unsqueeze(0).expand(B,L,-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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

def get_tmix_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return TimeMixState(
        torch.zeros([B, config.dim_att // config.head_size, config.head_size, config.head_size], dtype=x.dtype, device=x.device, requires_grad=requires_grad), 
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
    )

@dataclass
class LLMOutput:
    logits: torch.FloatTensor = None
    model_state: ModelState = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    post_attention_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    student_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    student_post_attention_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

class TMix_qwen2(nn.Module):
    def get_default_state_factory(self): return get_tmix_default_state

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len

        self.head_dim = config.head_size

        self.hidden_size = config.n_embd
        self.num_heads = config.dim_att // self.head_dim
        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads > 0 else self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # self.max_position_embeddings = config.max_position_embeddings
        # self.rope_theta = config.rope_theta
        # self.is_causal = True
        # self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # self.rotary_emb = Qwen2RotaryEmbedding(
        #     self.head_dim,
        #     max_position_embeddings=config.rope.max_seqlen,
        #     base=config.rope.base,
        # )

    def forward(self, x, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, L, D = x.size()
        QH = self.num_heads
        KVH = self.num_key_value_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        wkv_state = last_state.wkv_state

        # handle recurrent inference via maintaining a kv cache
        # if not self.training:
        #     new_kv_cache = torch.stack([k, v], dim=0)
        #     wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
        #     k, v = wkv_state.unbind(0)
        #     k, v = k.contiguous(), v.contiguous()

        is_causal = q.size(1)==k.size(1)

        q = q.view(B,L,QH,-1).transpose(1,2)
        k = k.view(B,L,KVH,-1).transpose(1,2)
        v = v.view(B,L,KVH,-1).transpose(1,2)

        #q, k = apply_rotary_embedding(q, k, shared.angles)
        #kv_seq_len, position_ids = L, torch.arange(L, dtype=torch.int, device=v.device).view(1, L).expand(B, L)
        #cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        cos, sin = shared.angles.unbind(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if output_attentions:
            attn_weights = (q * (self.head_dim ** -0.5)) @ k.mT

            #y = nn.functional.softmax(attn_weights + causal_mask, dim=-1, dtype=torch.float32).to(q.dtype) @ v
            #attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            #y = torch.matmul(attn_weights, v)

            # NOTE - we are outputting the non-softmaxed attention weights, just with exp() maxed to 1.0 since we're comparing against pre-normalized output of linear attention
            # upcast attention to fp32
            causal_mask = torch.full([L, L], fill_value=-torch.inf, device=attn_weights.device, dtype=attn_weights.dtype).triu(1)

            attn_weights = nn.functional.softmax(attn_weights + causal_mask, dim=-1, dtype=torch.float32).to(q.dtype)

            #attn_weights = attn_weights.tril()
            #attn_weights = (attn_weights - attn_weights.max() + causal_mask).exp()
            #attn_weights = (attn_weights - torch.max(attn_weights, dim=-1, keepdim=True).values + causal_mask).exp()
        else:
            attn_weights = torch.empty(0, device=x.device)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B,L,D)
        y = self.o_proj(y)
        return y, TimeMixState(wkv_state, last_state.shift_state), attn_weights

class TMix_qwen2rwkv(TMix_qwen2):
    """
    Qwen2 RWKV-6cSimple attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    and adds RWKV specific weights for tokenshift, decay, time_first, and the final layernorm.
    """

    def __init__(self, config:Transformer_Config, layer_id):
        super().__init__(config, layer_id)

        n_layer = config.n_layer
        n_embd = self.hidden_size
        dim_att = self.hidden_size
        layer_id = self.layer_id

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            # self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            ddd = torch.zeros(1, 1, n_embd)
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_k = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_v = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_w = nn.Parameter(torch.zeros_like(ddd))

            D_MIX_LORA = 32 if n_embd < 4096 else 64
            self.time_maa_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            # # per-head RWKV-6
            # H = self.num_heads
            # # fancy time_decay
            # decay_speed = torch.ones(H)
            # for h in range(H):
            #     decay_speed[h] = -6 + 5 * (h / max(H - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            # self.time_decay = nn.Parameter(decay_speed)
            # #self.time_decay = nn.Parameter(torch.empty(H)).uniform_(-8, -7)
            # D_DECAY_LORA = 64 if n_embd < 4096 else 128
            # self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            # self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, H).uniform_(-0.01, 0.01))

            # RWKV-6
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))
            D_DECAY_LORA = 64 if n_embd < 4096 else 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, dim_att).uniform_(-0.01, 0.01))
            # tmp = torch.zeros(dim_att)
            # for n in range(dim_att):
            #     zigzag = ((n + 1) % 3 - 1) * 0.1
            #     tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag
            # self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.ln_x = nn.LayerNorm(dim_att)

    def segsum(self, w_log): # B H L 1
        w_log_cumsum = torch.cumsum(w_log, dim=-2) # (B, H, L, 1)
        w_mask = torch.exp((w_log_cumsum - w_log_cumsum.mT).tril()).tril() # (B, H, L, L)
        return w_mask

    def forward(self, x, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        bsz, q_len, hidden_dim = x.size()

        dxprev = torch.nn.functional.pad(x, (0, 0, 1, -1)) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(bsz*q_len, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), bsz, q_len, hidden_dim)

        mr, mk, mv, mw = xxx.unbind(dim=0)
        #mr, mk, mv = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)

        query_states = self.q_proj(xr)
        key_states = self.k_proj(xk)
        value_states = self.v_proj(xv)
        #decay_states = self.time_decay.view(1,1,H,1).expand(bsz,q_len,H,1).contiguous()
        decay_states = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(query_states.dtype)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        decay_states = decay_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #dropout_rate = 0.0 if not self.training else self.attention_dropout

        decay_states_log = -decay_states.float().exp()
        decay_states_log = decay_states_log.clamp(-5) # FIXME - is this necessary?
        key_states = (key_states * (1 - decay_states_log.exp())).to(key_states.dtype)

        # kv_seq_len = key_states.shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)#, position_ids)
        #cos, sin = shared.angles.unbind(0)
        #query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        # # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # # therefore the input hidden states gets silently casted in float32. Hence, we need
        # # cast them back in float16 just to be sure everything works as expected.
        # input_dtype = query_states.dtype
        # if input_dtype == torch.float32:
        #     if torch.is_autocast_enabled():
        #         target_dtype = torch.get_autocast_gpu_dtype()
        #     # Handle the case where the model is quantized
        #     elif hasattr(self.config, "_pre_quantization_dtype"):
        #         target_dtype = self.config._pre_quantization_dtype
        #     else:
        #         target_dtype = self.q_proj.weight.dtype

        #     logger.warning_once(
        #         f"The input hidden states seems to be silently casted in float32, this might be related to"
        #         f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #         f" {target_dtype}."
        #     )

        #     query_states = query_states.to(target_dtype)
        #     key_states = key_states.to(target_dtype)
        #     value_states = value_states.to(target_dtype)

        # decay_states_log.view is to match fla_chunk_simple_gla's requirements
        #print("layer", self.layer_id, "pre ", bool(query_states.isnan().any()), bool(key_states.isnan().any()), bool(value_states.isnan().any()), bool(decay_states_log.isnan().any()))
        #o = chunk_simple_gla(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), scale)

        #print("layer", self.layer_id, "post", bool(query_states.isnan().any()), bool(key_states.isnan().any()), bool(value_states.isnan().any()), bool(decay_states_log.isnan().any()))

        if not output_attentions:
            attn_weights = torch.empty(0, device=x.device)

            #attn_output = fla_chunk_simple_gla(query_states, key_states, value_states, decay_states_log.view(bsz, self.num_heads, q_len))[0]
            attn_output = fla_chunk_gla(query_states, key_states, value_states, decay_states_log)[0]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.ln_x(attn_output)
            attn_output = self.o_proj(attn_output)
        else:
            attn_weights = (query_states * (key_states.size(-1) ** -0.5)) @ key_states.mT

            decay_states_log = decay_states_log.mean(-1, keepdim=True)
            attn_weights = attn_weights.float() * self.segsum(decay_states_log.float()) # NOTE - without the explicit cast to float ddp mismatched deepspeed here

            attn_weights = attn_weights.to(query_states.dtype)
            attn_output = torch.empty(0, device=x.device)
    
        return attn_output, TimeMixState(last_state.wkv_state, last_state.shift_state), attn_weights #, past_key_value
    
def get_cmix_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return ChannelMixState(
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
    )

class CMix_qwen2(nn.Module):
    def get_default_state_factory(self): return get_cmix_default_state

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.hidden_size = config.n_embd
        self.intermediate_size = config.dim_ffn
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = torch.nn.SiLU() #ACT2FN[config.hidden_act]

    def forward(self, x, last_model_state:ModelState):
        last_state = last_model_state.block_states[self.layer_id].channel_mix_state
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), last_state

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config:TrainerCLI_Config, layer_id:int):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        self.input_layernorm = Qwen2RMSNorm(args.n_embd, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(args.n_embd, eps=args.rms_norm_eps)

        cmix = CMix_qwen2(args, layer_id)

        if args.attention_type == 'rwkv':
            self.self_attn = TMix_qwen2rwkv(args, layer_id)
        else:
            self.self_attn = TMix_qwen2(args, layer_id)
        self.default_time_mix_state_factory = self.self_attn.get_default_state_factory() if hasattr(self.self_attn, 'get_default_state_factory') else lambda x, c, r: TimeMixState()

        self.teacher_attn = None
        if config.train is not None:
            if config.train.attention_distillation_stage in (1, 2):
                self.teacher_attn = TMix_qwen2(args, layer_id)
        
        self.default_channel_mix_state_factory = cmix.get_default_state_factory() if hasattr(cmix, 'get_default_state_factory') else lambda x, c, r: ChannelMixState()
        self.mlp = cmix

    def forward(self, x:Tensor, last_model_state:ModelState, shared:Shared, output_attentions:bool, output_post_attention_hidden_states:bool):
        s = last_model_state
        if self.teacher_attn is not None:
            dx, last_timemix_state, attentions = self.teacher_attn(self.input_layernorm(x), s, shared, output_attentions)
            student_dx, student_last_timemix_state, student_attentions = self.self_attn(self.input_layernorm(x), s, shared, output_attentions)
        else:
            dx, last_timemix_state, attentions = self.self_attn(self.input_layernorm(x), s, shared, output_attentions)
            student_dx, student_last_timemix_state, student_attentions = None, None, None
        if output_post_attention_hidden_states:
            post_attention_hidden_states = dx
            student_post_attention_hidden_states = student_dx
        else:
            post_attention_hidden_states = torch.empty(0, device=x.device)
            student_post_attention_hidden_states = torch.empty(0, device=x.device)
            
        x = x + dx
        dx, last_chanmix_state = self.mlp(self.post_attention_layernorm(x), s)
        x = x + dx
        return x, s, attentions, post_attention_hidden_states, student_attentions, student_post_attention_hidden_states

def ckpt(block:Qwen2DecoderLayer, *block_args):
    if block.training and block.config.train.grad_cp == 1 and 'fsdp' not in block.config.train.strategy: # FSDP has its own checkpointing wrapper
        #if "deepspeed" in block.config.train.strategy:
        #    results = deepspeed.checkpointing.checkpoint(block, *block_args)
        #else:
        # NOTE - both deepspeed.checkpointing.checkpoint and use_reentrant=True failed miserably (bad loss) when used in conjunction with requires_grad=False params and grad_cp with deepspeed
        results = torch.utils.checkpoint.checkpoint(block, *block_args, use_reentrant=False)
    else:
        results = block(*block_args)
    return results

class Qwen2Decoder(nn.Module):
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        self.shared = Shared()

        self.embed_tokens = nn.Embedding(args.vocab_size, args.n_embd, args.vocab_padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_id) for layer_id in range(args.n_layer)]
        )
        self.norm = Qwen2RMSNorm(args.n_embd, eps=args.rms_norm_eps)

    def forward_preamble(self, x, last_model_state:ModelState|None = None, ):
        config : Transformer_Config = self.config.model

        B, T, C = x.size()

        shared = self.shared
        if config.rope is not None and shared.angles.size(0) == 0:
            shared.angles = generate_rotary_embedding(config.ctx_len, config.head_size, config.rope.base * config.rope.rebase, config.rope.rescale).to(self.norm.weight)

        assert (shared.angles.size(0) == 0 or T <= shared.angles.size(0)) or (shared.bias_mask.size(0) == 0 or T <= shared.bias_mask.size(0))

        # might need to be true in the future for BPTT support
        requires_grad = self.training
        if last_model_state is None:
            last_model_state = ModelState()
            for layer_id in range(config.n_layer):
                layer = self.layers[layer_id]
                last_model_state.block_states.append(BlockState(
                    layer.default_time_mix_state_factory(x, config, requires_grad),
                    layer.default_channel_mix_state_factory(x, config, requires_grad),
                ))

        return last_model_state

    def forward(self, token_ids:Tensor|list, last_model_state:ModelState|None = None, output_hidden_states:bool=False, output_attentions:bool=False, output_post_attention_hidden_states:bool=False):
        config : Transformer_Config = self.config.model
        if isinstance(token_ids, Tensor):
            B, T = token_ids.size()
        else:
            B = 1
            T = len(token_ids)
            token_ids = torch.tensor(token_ids, device=self.embed_tokens.weight.device, dtype=torch.long, requires_grad=False)[None, :]

        x = self.embed_tokens(token_ids)

        last_model_state = self.forward_preamble(x, last_model_state)

        hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs = (), (), ()
        student_hidden_states_outputs, student_attentions_outputs, student_post_attention_hidden_states_outputs = (), (), ()
        if output_hidden_states:
            hidden_states_outputs += (x,)
            student_hidden_states_outputs += (x,)
        for decoder_layer in self.layers:
            x, s, attentions, post_attention_hidden_states, student_attentions, student_post_attention_hidden_states = ckpt(decoder_layer, x, last_model_state, self.shared, output_attentions, output_post_attention_hidden_states)
            hidden_states_outputs += (x,)
            student_hidden_states_outputs += (x,)
            if output_attentions:
                attentions_outputs += (attentions,)
                student_attentions_outputs += (student_attentions,)
            if output_post_attention_hidden_states:
                post_attention_hidden_states_outputs += (post_attention_hidden_states,)
                student_post_attention_hidden_states_outputs += (student_post_attention_hidden_states,)

        x = self.norm(x)
        return LLMOutput(x, last_model_state, hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs, student_attentions_outputs, student_post_attention_hidden_states_outputs)
        #return x, last_model_state, hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs # FIXME - not updating state at all

class Model_qwen2(nn.Module): # Qwen2CausalLM
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        if args.dim_att <= 0:
            args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

        self.model = None

        #self.configure_model()

    def configure_model(self):
        if self.model is not None: 
            return

        self.model = Qwen2Decoder(self.config)

        self.lm_head = nn.Linear(self.config.model.n_embd, self.config.model.vocab_size, bias=False)

    def forward(self, token_ids:Tensor|list, last_model_state:ModelState|None = None, output_hidden_states:bool=False, output_attentions:bool=False, output_post_attention_hidden_states:bool=False):
        #print("teacher q min, max", float(self.model.layers[0].self_attn.q_proj.weight.min()), float(self.model.layers[0].self_attn.q_proj.weight.max()))
        results = self.model(token_ids, last_model_state=last_model_state, output_hidden_states=output_hidden_states, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)
        results.logits = self.lm_head(results.logits)
        return results
    
    def get_optim_groups(self):
        # separates groups for weight decay and non-weight decay

        train_config = self.config.train

        if train_config.attention_distillation_stage in (1, 2):
            self.requires_grad_(False)
            for decoder_layer in self.model.layers:
                if train_config.attention_distillation_stage == 1:
                    decoder_layer.self_attn.time_maa_x.requires_grad_(True)
                    decoder_layer.self_attn.time_maa_r.requires_grad_(True)
                    decoder_layer.self_attn.time_maa_k.requires_grad_(True)
                    #decoder_layer.self_attn.time_maa_w.requires_grad_(True) # FIXME!!! we left this out!
                    decoder_layer.self_attn.time_maa_w1.requires_grad_(True)
                    decoder_layer.self_attn.time_maa_w2.requires_grad_(True)
                    decoder_layer.self_attn.time_decay.requires_grad_(True)
                    decoder_layer.self_attn.time_decay_w1.requires_grad_(True)
                    decoder_layer.self_attn.time_decay_w2.requires_grad_(True)
                    # FIXME - wow we removed q, k here by accident and it.. helped??!?!
                elif train_config.attention_distillation_stage == 2:
                    decoder_layer.self_attn.requires_grad_(True)

        # FIXME - remove these for full training
        # for decoder_layer in self.model.layers:
        #     decoder_layer.post_attention_layernorm.requires_grad_(False)
        #     decoder_layer.mlp.requires_grad_(False)
        # self.model.embed_tokens.requires_grad_(False)
        # self.model.norm.requires_grad_(False)
        # self.lm_head.requires_grad_(False)

        # JIT at last minute
        for decoder_layer in self.model.layers:
            decoder_layer.self_attn = TJIT(decoder_layer.self_attn)
            decoder_layer.mlp = TJIT(decoder_layer.mlp)

        lr_decay = set()
        lr_1x = set()
        lr_tiny = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            #if 'lm_head.weight' in n:
            #    lr_tiny.add(n)
            #el
            if (len(p.squeeze().shape) >= 2) and (train_config.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        param_dict = {n: p for n, p in self.named_parameters()}
        param_check = list(lr_decay) + list(lr_1x) + list(lr_tiny)
        #if not train_config.load_partial and (train_config.teacher is None or train_config.teacher.attention_distillation_stage ==3):
        #    assert sorted(param_dict) == sorted(param_check)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_tiny = sorted(list(lr_tiny))
        
        print('decay', lr_decay, '\n')
        print('1x', lr_1x, '\n')
        print('tiny', lr_tiny, '\n')

        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name':'lr_1x'},
        ]
        if len(lr_tiny) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_tiny], "weight_decay": train_config.weight_decay, "my_lr_scale": 0.01, 'name':'lr_tiny'}]
        if len(lr_decay) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_decay'}]

        return optim_groups    

    def _init_weights(self, module):
        std = 0.02 #self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def init_all_weights(self):
        self.apply(self._init_weights)
        for n, p in self.named_parameters():
            requires_grad_temp = p.requires_grad
            p.requires_grad_(False)
            if n.endswith('.ln_x.weight'):
                layer_scale = (1+int(n.split('.')[2])) / self.config.model.n_layer
                print('.ln_x.weight layer', int(n.split('.')[2]), "scale", (layer_scale ** 0.7))
                p *= 0.0
                p += (layer_scale ** 0.7)
            p.requires_grad = requires_grad_temp
