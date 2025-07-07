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

ATTENTION_TYPE = os.environ["RWKV_ATTENTION_TYPE"]
if ATTENTION_TYPE == 'rwkv6':
    from cuda import rwkv6_cuda
elif 'gla' in ATTENTION_TYPE:
    from fla.ops.gla.chunk import chunk_gla
    from fla.ops.gla.fused_recurrent import fused_recurrent_gla
elif 'rwkv7' in ATTENTION_TYPE or ATTENTION_TYPE == 'rwkv6_wind_backstepping_longhead':
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    CTX_LEN = int(os.environ["RWKV_CTXLEN"])

    from torch.utils.cpp_extension import load

    if ATTENTION_TYPE == 'rwkv7':
        ATTENTION_TYPE = 'rwkv7_wind_triton_bighead'
    # if torch.version.hip and 'fla_' not in ATTENTION_TYPE:
    #     ATTENTION_TYPE ='rwkv7_wind_triton_bighead'

    if ATTENTION_TYPE == 'rwkv7_fla_chunk':
        from fla.ops.rwkv7.chunk import chunk_rwkv7
        def RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, a, b) -> torch.Tensor:
            log_w = -log_neglog_w.float().exp()
            return chunk_rwkv7(r=r, w=log_w, k=k, v=v, a=a, b=b, output_final_state=False)[0].to(dtype)
    elif ATTENTION_TYPE == 'rwkv7_fla_fused_recurrent':
        from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
        def RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, a, b, s0=None) -> torch.Tensor:
            log_w = -log_neglog_w.float().exp()
            return fused_recurrent_rwkv7(r, log_w, k, v, a, b, initial_state=s0, output_final_state=True)
    elif ATTENTION_TYPE == 'rwkv7_wind_triton':
        from rwkv7_attn_triton import TritonRWKV7
        def RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, a, b, s0=None) -> torch.Tensor:
            B,T,H,C = r.shape
            if s0 is None:
                s0 = torch.zeros(B,H,C,C,dtype=r.dtype,device=r.device)
            return TritonRWKV7.apply(log_neglog_w, r, k, v, a, b, s0, 'fp32')
    elif ATTENTION_TYPE == 'rwkv7_wind_triton_bighead':
        from rwkv7_attn_triton_bighead import TritonRWKV7
        def RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, a, b, s0=None) -> torch.Tensor:
            B,T,H,C = r.shape
            if s0 is None:
                s0 = torch.zeros(B,H,C,C,dtype=r.dtype,device=r.device)
            return TritonRWKV7.apply(log_neglog_w, r, k, v, a, b, s0, 'fp32')
    elif ATTENTION_TYPE in ('rwkv7_wind_cuda', 'rwkv7_wind_cuda_full'):
        CHUNK_LEN = 16

        flags = [f'-D_C_={HEAD_SIZE}', "-O3"]
        if torch.cuda.is_available():
            if torch.version.hip:
                flags += ["--save-temps"]
            else:
                flags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

        if ATTENTION_TYPE == 'rwkv7_wind_cuda_full':
            sources=['rwkv_cuda_wind/wind_rwkv7_full.cu']
        else:
            sources=['rwkv_cuda_wind/wind_rwkv7.cpp', 'rwkv_cuda_wind/wind_rwkv7.cu']

        load(name="wind", sources=sources, is_python_module=False, verbose=True, extra_cuda_cflags=flags)

        class WindRWKV7(torch.autograd.Function):
            @staticmethod
            def forward(ctx,w,q,k,v,a,b):
                B,T,H,C = w.shape
                s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
                assert T%16 == 0
                assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
                w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
                y = torch.empty_like(v)
                sT = torch.empty_like(s0)
                s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
                torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
                ctx.save_for_backward(w,q,k,v,a,b,s)
                return y
            
            @staticmethod
            def backward(ctx,dy):
                w,q,k,v,a,b,s = ctx.saved_tensors
                B,T,H,C = w.shape
                dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
                assert all(i.dtype==torch.bfloat16 for i in [dy])
                dy,dsT = [i.contiguous() for i in [dy,dsT]]
                dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
                torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
                return dw,dq,dk,dv,da,db

        def RUN_CUDA_RWKV7g(q,w,k,v,a,b) -> torch.Tensor:
            B,T,H,N = q.shape
            return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,-1)
            
    elif ATTENTION_TYPE in ['rwkv6_wind_backstepping_longhead', 'rwkv7_wind_backstepping_longhead', 'rwkv7_wind_singlestepping_longhead', 'rwkv7_wind_replaystepping_longhead']:
        if ATTENTION_TYPE == 'rwkv7_wind_singlestepping_longhead':
            CHUNK_LEN = 1
        else:
            CHUNK_LEN = 16

        batchsz_times_heads_estimate = 8*64
        head_size = HEAD_SIZE

        device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if 'AMD' in device_props.name:
            value_chunk_size = 16
        else:
            value_chunk_size = 64
            if torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count >= batchsz_times_heads_estimate * head_size / 32:
                value_chunk_size = 32

        flags = [f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-O3"]
        if torch.cuda.is_available():
            if torch.version.hip:
                flags += ["--save-temps", '-ffast-math', '-DAMD']
            else:
                flags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

        flags += [f'-D_K_={value_chunk_size}']
        sources = [f'rwkv_cuda_wind/{ATTENTION_TYPE[11:]}.cu', f'rwkv_cuda_wind/{ATTENTION_TYPE[11:]}.cpp']

        load(name=ATTENTION_TYPE[6:], sources=sources, is_python_module=False, verbose=True, extra_cuda_cflags=flags)
        if ATTENTION_TYPE in ['rwkv6_wind_backstepping_longhead', 'rwkv7_wind_backstepping_longhead']:
            op = torch.ops.wind_backstepping_longhead
            assert hasattr(op, 'forward'), 'Requires a load kernel from load_backstepping(head_size)'
        elif ATTENTION_TYPE == 'rwkv7_wind_singlestepping_longhead':
            op = torch.ops.wind_singlestepping_longhead
            assert hasattr(op, 'forward'), 'Requires a load kernel from load_singlestepping(head_size)'
        elif ATTENTION_TYPE == 'rwkv7_wind_replaystepping_longhead':
            op = torch.ops.wind_replaystepping_longhead
            assert hasattr(op, 'forward'), 'Requires a load kernel from load_replaystepping(head_size)'

        class RWKV7_longhead(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q,w,k,v,a,b,s0):
                B,T,H,C = w.shape
                assert T%CHUNK_LEN == 0
                if not torch.compiler.is_compiling():
                    #assert hasattr(op, 'forward'), 'Requires a load kernel from load_replaystepping(head_size)'
                    assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
                    w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
                    #assert all(i.is_contiguous() for i in [w,q,k,v,a,b,s0])
                    assert all(i.shape == w.shape for i in [w,q,k,v,a,b])
                    assert list(s0.shape) == [B,H,C,C]
                B,T,H,C = w.shape
                y = torch.empty_like(v)
                sT = torch.empty_like(s0)
                if any(i.requires_grad for i in [w,q,k,v,a,b,s0]):
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                else:
                    s = sa = None
                op.forward(w,q,k,v,a,b, s0,y,s,sa,sT)
                ctx.save_for_backward(w,q,k,v,a,b,s,sa)
                return y, sT
            @staticmethod
            def backward(ctx, dy, dsT):
                w,q,k,v,a,b,s,sa = ctx.saved_tensors
                B,T,H,C = w.shape
                if not torch.compiler.is_compiling():
                    assert all(i.dtype==torch.bfloat16 for i in [dy,dsT])
                    assert all(i.is_contiguous() for i in [dy,dsT])

                dv,ds0 = [torch.empty_like(x) for x in [v,dsT]]
                dw,dq,dk,da,db = [torch.zeros(B,T,H,C, device=w.device) for i in range(5)]
                op.backward(w,q,k,v,a,b, dy,s,sa,dsT, dw,dq,dk,dv,da,db,ds0)
                return dq,dw,dk,dv,da,db,ds0
            
        def RUN_CUDA_RWKV7g(q,log_neglog_w,k,v,a=None,b=None,s0=None) -> torch.Tensor:
            B,T,H,N = q.shape
            if a is None: a = torch.zeros(B,T,H,HEAD_SIZE, dtype=torch.bfloat16, device=log_neglog_w.device)
            if b is None: b = torch.zeros(B,T,H,HEAD_SIZE, dtype=torch.bfloat16, device=log_neglog_w.device)
            if s0 is None: s0 = torch.zeros(B,H,HEAD_SIZE,HEAD_SIZE, dtype=torch.bfloat16,device=log_neglog_w.device)
            y, s_out = RWKV7_longhead.apply(q,log_neglog_w,k,v,a,b,s0)
            return y.view(B,T,-1), s_out
                   
    elif ATTENTION_TYPE in ['rwkv7_wind_backstepping', 'rwkv7_wind_backstepping_smallhead', 'rwkv7_wind_backstepping_bighead']:
        CHUNK_LEN = 16

        batchsz_times_heads_estimate = 8*64
        head_size = HEAD_SIZE

        flags = [f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-O3"]
        if torch.cuda.is_available():
            if torch.version.hip:
                flags += ["--save-temps", '-ffast-math', '-DAMD']
            else:
                flags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

        #if ATTENTION_TYPE == 'rwkv7_wind_backstepping_full':
        #    sources=[f'rwkv_cuda_wind/backstepping_f32_{VERSION}_full.cu']
        #else:
        if ATTENTION_TYPE == 'rwkv7_wind_backstepping_smallhead':
            VERSION = 1
        elif ATTENTION_TYPE == 'rwkv7_wind_backstepping_bighead':
            VERSION = 2
        else:
            VERSION = 1 if HEAD_SIZE < 128 else 2

        sources=[f'rwkv_cuda_wind/backstepping_f32_{VERSION}.cu', f'rwkv_cuda_wind/backstepping_f32_cpp.cu']

        #load(name="wind_backstepping", sources=[f'rwkv_cuda_wind/backstepping_f32_{VERSION}_full.cu'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
        load(name="wind_backstepping", sources=sources, is_python_module=False, verbose=True, extra_cuda_cflags=flags)


        class WindBackstepping(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w,q,k,v,z,b):
                B,T,H,C = w.shape 
                assert T%CHUNK_LEN == 0
                assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
                assert all(i.shape == w.shape for i in [w,q,k,v,z,b])
                w,q,k,v,z,b = [i.contiguous() for i in [w,q,k,v,z,b]]
                y = torch.empty_like(v)
                s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
                ctx.save_for_backward(w,q,k,v,z,b,s,sa)
                return y
            @staticmethod
            def backward(ctx, dy):
                assert dy.dtype == torch.bfloat16
                dy = dy.contiguous()
                w,q,k,v,z,b,s,sa = ctx.saved_tensors
                dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
                torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
                return dw,dq,dk,dv,dz,db

        def RUN_CUDA_RWKV7g(q,log_neglog_w,k,v,a,b) -> torch.Tensor:
            B,T,H,N = q.shape
            return WindBackstepping.apply(log_neglog_w,q,k,v,a,b).view(B,T,-1)
    elif ATTENTION_TYPE == 'rwkv7_cuda':
        DTYPE = torch.bfloat16
        XTYPE = torch.float
        T = CTX_LEN
        CHUNK_LEN = 16

        flags = [f'-D_N_={HEAD_SIZE}', f'-D_T_={T}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-O3"]
        if torch.cuda.is_available():
            if torch.version.hip:
                flags += ["--save-temps"]
            else:
                flags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

        load(name="wkv7g", sources=["cuda/wkv7g_op.cpp", f"cuda/wkv7g_v1.cu"], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
        class WKV_7g(torch.autograd.Function):
            @staticmethod
            def forward(ctx, r, w, k, v, a, b, s):
                with torch.no_grad():
                    B, T, H, V = v.size()
                    B, T, H, R = r.size()
                    #N = C // H
                    A = T // CHUNK_LEN
                    #assert N == C // H
                    assert T % CHUNK_LEN == 0
                    assert all(i.dtype == DTYPE for i in [r,w,k,v,a,b])
                    r,w,k,v,a,b = [i.contiguous() for i in [r,w,k,v,a,b]]
                    ctx.B = B
                    ctx.T = T
                    ctx.R = R
                    ctx.V = V
                    ctx.H = H
                    y = torch.empty((B, T, V), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    saa = torch.empty((B, T, H, R), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    sss = torch.empty((B, H, A, R, R), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv7g.forward(B, T, V*H, H, r, w, k, v, a, b, y, saa, sss)
                    ctx.save_for_backward(r, w, k, v, a, b, saa, sss)
                    return y
            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    B = ctx.B
                    T = ctx.T
                    R = ctx.R
                    V = ctx.V
                    H = ctx.H
                    A = T // CHUNK_LEN
                    assert gy.dtype == DTYPE
                    gy = gy.contiguous()
                    r, w, k, v, a, b, saa, sss = ctx.saved_tensors
                    gr = torch.empty((B, T, R), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                    gw = torch.empty((B, T, R), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                    gk = torch.empty((B, T, R), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                    gv = torch.empty((B, T, V), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                    ga = torch.empty((B, T, R), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gb = torch.empty((B, T, R), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    zzz = torch.empty((B, H, A-1, N, N), device=gy.device, dtype=XTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv7g.backward(B, T, V*H, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb)
                    del saa
                    del sss
                    del zzz
                    return (gr, gw, gk, gv, ga, gb)
        def RUN_CUDA_RWKV7g(r, w, k, v, a, b) -> torch.Tensor:
            return WKV_7g.apply(r, w, k, v, a, b)  
    else:
        assert False, 'bad attention type specified'
elif ATTENTION_TYPE == 'softpick':
    from flash_softpick_attn import parallel_softpick_attn
else:
    assert False, 'bad attention type specified'

def rms_norm(hidden_states, eps = 1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen3
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
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
    B = q.size(0)
    if unsqueeze_dim == 2:
        L = q.size(1)
    else:
        L = q.size(2)
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

class TMix_qwen3(nn.Module):
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

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # self.rotary_emb = Qwen3RotaryEmbedding(
        #     self.head_dim,
        #     max_position_embeddings=config.rope.max_seqlen,
        #     base=config.rope.base,
        # )

    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
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

        q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)
        k = self.k_norm(k.view(B,L,KVH,-1)).transpose(1,2)
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
        y = y.transpose(1,2).reshape(B,L,-1)
        y = self.o_proj(y)
        return y, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights

class TMix_qwen3softpick(TMix_qwen3):
    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, L, D = x.size()
        QH = self.num_heads
        KVH = self.num_key_value_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        wkv_state = last_state.wkv_state

        q = self.q_norm(q.view(B,L,QH,-1))
        k = self.k_norm(k.view(B,L,KVH,-1))
        v = v.view(B,L,KVH,-1)

        cos, sin = shared.angles.unbind(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, L, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, L, self.num_heads, -1)
        v = v.view(B, L, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, L, self.num_heads, -1)

        assert not output_attentions
        attn_weights = torch.empty(0, device=x.device)

        y = parallel_softpick_attn(q, k, v)
        y = y.reshape(B,L,-1)
        y = self.o_proj(y)
        return y, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights
    
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
    
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, create_mask
from functools import lru_cache

SLIDING_WINDOW = 1024

def sliding_window_causal(score, b, h, q_idx, kv_idx):
    return torch.where((q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW), score, -float("inf"))

def sliding_window_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
    return causal_mask & window_mask

block_mask = None

class TMix_qwen3newatt(TMix_qwen3):
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)

        n_embd = config.n_embd

        ddd = torch.empty(1, 1, n_embd)
        self.time_maa_x = nn.Parameter(torch.empty_like(ddd))
        self.time_maa_k = nn.Parameter(torch.empty_like(ddd))
        self.time_maa_v = nn.Parameter(torch.empty_like(ddd))

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32

        lora_rank_tokenshift = config.lora_rank_tokenshift or calc_lora_rank(0.5, 1.8)
        #lora_rank_tokenshift = 32 if n_embd < 4096 else 64

        self.time_maa_w2 = nn.Parameter(torch.empty(2, lora_rank_tokenshift, n_embd))
        self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, lora_rank_tokenshift*self.time_maa_w2.size(0)))

        # self.k_proj2 = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj2 = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        args = self.config
        D_VALUE_LORA = max(args.n_embd // 16, 64)
        self.time_key_w1 = nn.Parameter(torch.zeros(args.n_embd, D_VALUE_LORA))
        self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, args.n_embd).uniform_(-0.01, 0.01))
        self.time_value_w1 = nn.Parameter(torch.zeros(args.n_embd, D_VALUE_LORA))
        self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, args.n_embd).uniform_(-0.01, 0.01))

        C = n_embd
        N = self.head_dim = config.head_size
        H = self.num_heads
        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)

        self.v0 = nn.Parameter(torch.empty(1,1,H*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))

    def reset_parameters(self):
        print("Called reset_parameters on TMix_qwen3newatt layer ", self.layer_id)

        module = self

        config = self.config

        n_layer = config.n_layer
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_id

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            ddd = torch.zeros(1, 1, n_embd)
            module.time_maa_x.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            module.time_maa_k.zero_()
            module.time_maa_v.zero_()

            module.time_maa_w2.uniform_(-0.01, 0.01)
            module.time_maa_w1.zero_()

            if layer_id > 0:
                module.v0.copy_(1.0)
                module.v1.zero_()
                ortho_init(module.v2, 0.1)

    # def reset_parameters_after_load(self):
    #     print("Called reset_parameters_after_load on TMix_qwen3newatt layer ", self.layer_id)

    #     module = self

    #     dim_att = self.num_heads * self.head_dim

    #     with torch.no_grad():
    #         module.q_proj.weight.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
    #         module.k_proj.weight.uniform_(-0.05/(self.hidden_size**0.5), 0.05/(self.hidden_size**0.5))
    #         module.v_proj.weight.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
    #         # ortho_init(module.q_proj.weight, 0.1)
    #         # ortho_init(module.k_proj.weight, 0.1)
    #         # ortho_init(module.v_proj.weight, 0.1)
    #         module.o_proj.weight.zero_()

    #         module.q_norm.weight.copy_(1.0)
    #         module.k_norm.weight.copy_(1.0)

    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, L, D = x.size()
        QH = self.num_heads
        KVH = self.num_key_value_heads

        # dfrozen_residualprev = F.pad(frozen_residual, (0, 0, 1, -1)) - frozen_residual
        # dresidual_first = F.pad(residual_first, (0, 0, 1, -1)) - residual_first

        # xxx = frozen_residual + dfrozen_residualprev * self.time_maa_x
        # xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*L, self.time_maa_w2.size(0), -1).transpose(0, 1)
        # xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, L, D)

        # mk, mv = xxx.unbind(dim=0)
        # xk = frozen_residual + dfrozen_residualprev * (self.time_maa_k + mk)
        # xv = frozen_residual + dfrozen_residualprev * (self.time_maa_v + mv)
        
        # xk = xv = frozen_residual
        # # if self.layer_id >= self.config.n_layer - 1: #self.config.preserve_last_n_layers // 2:
        # # xk = xv = x
        # # #xk = x
        # # #xv = x
        # # xk = xv = frozen_residual

        # xk = xk + torch.tanh(xk @ self.time_key_w1) @ self.time_key_w2
        # xv = xv + torch.tanh(xv @ self.time_value_w1) @ self.time_value_w2

        #x = xv = xk = frozen_residual
        #xv = xk = frozen_residual
        #xk = x
        #xq = frozen_residual
        #xk = frozen_residual
        #xv = residual_first
        #xv = frozen_residual
        xq = x
        xk = x
        xv = x

        #if self.layer_id != 0:
        #    xv = residual_first

        q = self.q_proj(xq)
        k = self.k_proj(xk)
        v = self.v_proj(xv)

        wkv_state = last_state.wkv_state

        # handle recurrent inference via maintaining a kv cache
        # if not self.training:
        #     new_kv_cache = torch.stack([k, v], dim=0)
        #     wkv_state = torch.cat([wkv_state, new_kv_cache], dim=-2)
        #     k, v = wkv_state.unbind(0)
        #     k, v = k.contiguous(), v.contiguous()

        is_causal = q.size(1)==k.size(1)

        q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)
        k = self.k_norm(k.view(B,L,KVH,-1)).transpose(1,2)
        v = v.view(B,L,KVH,-1).transpose(1,2)

        cos, sin = shared.angles.unbind(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # if self.layer_id == 0:
        #     v_first = v
        # else:
        #     v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2).view_as(v)

        attn_weights = torch.empty(0, device=x.device)

        # global block_mask
        # if block_mask is None:
        #     block_mask = create_block_mask(mask_mod=sliding_window_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L, device=q.device)
        # y = flex_attention(query=q, key=k, value=v, score_mod=sliding_window_causal, block_mask=block_mask)
        # global block_mask
        # if block_mask is None:
        #     block_mask = create_mask(mod_fn=sliding_window_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L, device=q.device)
        #     print(block_mask)
        #     #block_mask[:, 0] = True
        # y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=block_mask)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B,L,-1)
        y = self.o_proj(y)
        return y, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights

class TMix_qwen3swa(TMix_qwen3):
    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        if SLIDING_WINDOW >= x.size(-2):
            return super().forward(x, residual_first, frozen_residual, reset_mask, v_first, last_model_state, shared, output_attentions)

        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, L, D = x.size()
        QH = self.num_heads
        KVH = self.num_key_value_heads

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        wkv_state = last_state.wkv_state

        q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)
        k = self.k_norm(k.view(B,L,KVH,-1)).transpose(1,2)
        v = v.view(B,L,KVH,-1).transpose(1,2)
        
        cos, sin = shared.angles.unbind(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        assert not output_attentions
        attn_weights = torch.empty(0, device=x.device)

        global block_mask
        if block_mask is None:
            block_mask = create_mask(mod_fn=sliding_window_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L, device=q.device)
        #     temp_mask = torch.ones(L, L, dtype=torch.bool, device=q.device).tril(diagonal=0)
        #     attn_bias = torch.zeros(L, L, dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        #     block_mask = attn_bias.to(v.dtype)
        # #     # block_mask = torch.ones(L, L, dtype=torch.bool, device=q.device).tril()
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=block_mask)
        #y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=torch.ones(L, L, dtype=torch.bool, device=q.device).tril())

        #y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        # do sliding window by (inefficiently) subtracting off the part outside of the window, because we found implementation difference (off by 1/256th for bfloat16) in sdpa causal vs masked
        #if SLIDING_WINDOW < L:
        #    ym = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=torch.ones(L, L, dtype=torch.bool, device=q.device).tril(diagonal=-SLIDING_WINDOW))
        #    y = y - ym
        #if not (ym==y).all().item():
        #    print("diff", (y-ym).abs().max())
        #    assert False, 'bad attn diff'
        # L, S = q.size(-2), k.size(-2)
        # scale_factor = 1 / math.sqrt(q.size(-1))
        # attn_weight = (q.float() * scale_factor) @ k.transpose(-2, -1).float()
        # #attn_weight += block_mask
        # attn_weight = torch.softmax(attn_weight, dim=-1).to(v.dtype)
        # attn_weight = attn_weight.tril()
        # y = attn_weight @ v

        y = y.transpose(1,2)
        y = y.reshape(B,L,-1)
        y = self.o_proj(y)
        return y, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights

class TMix_qwen3doubleatt(TMix_qwen3):
    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        if SLIDING_WINDOW >= x.size(-2):
            return super().forward(x, residual_first, frozen_residual, reset_mask, v_first, last_model_state, shared, output_attentions)

        args = self.config
        if self.layer_id == args.n_layer - args.preserve_last_n_layers:
            x = torch.cat([x, x], dim=1)

        #print('layer_id', self.layer_id)

        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        B, L, D = x.size()
        L = L // 2
        QH = self.num_heads
        KVH = self.num_key_value_heads

        wkv_state = last_state.wkv_state

        w, x = x.view(B, 2, L, D).unbind(1)
        #print('w.shape', w.shape)

        # sliding window attention

        q = self.q_proj(w)
        k = self.k_proj(w)
        v = self.v_proj(w)

        is_causal = q.size(1)==k.size(1)

        q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)
        k = self.k_norm(k.view(B,L,KVH,-1)).transpose(1,2)
        v = v.view(B,L,KVH,-1).transpose(1,2)

        # q = self.q_proj(x)
        # q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)

        cos, sin = shared.angles.unbind(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.empty(0, device=x.device)

        global block_mask
        if block_mask is None:
            block_mask = create_mask(mod_fn=sliding_window_causal_mask, B=None, H=None, Q_LEN=L, KV_LEN=L, device=q.device)
        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=block_mask)
        y = y.transpose(1,2).reshape(B,L,-1)
        y_w = self.o_proj(y)

        # global attention

        q = self.q_proj(x)
        q = self.q_norm(q.view(B,L,QH,-1)).transpose(1,2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        q = q.to(v.dtype)

        y = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
        y = y.transpose(1,2).reshape(B,L,-1)
        y = self.o_proj(y)

        if self.layer_id < args.n_layer - 1:
            y = torch.cat([y_w, y], dim=1)
        #print('y.shape', y.shape)

        return y, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights
    
class TMix_qwen3rwkv6(TMix_qwen3):
    """
    Qwen3 RWKV-6cSimple attention module, following Qwen3 attention module. This module inherits from `Qwen3Attention`
    and adds RWKV specific weights for tokenshift, decay, time_first, and the final layernorm.
    """

    def __init__(self, config:Transformer_Config, layer_id):
        super().__init__(config, layer_id)
    
        n_layer = config.n_layer
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        self.qk_head_dim = self.head_dim
        layer_id = self.layer_id

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32

        if config.use_tokenshift:
            # self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            # self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            ddd = torch.empty(1, 1, n_embd)
            self.time_maa_x = nn.Parameter(torch.empty_like(ddd))
            self.time_maa_r = nn.Parameter(torch.empty_like(ddd))
            self.time_maa_k = nn.Parameter(torch.empty_like(ddd))
            self.time_maa_v = nn.Parameter(torch.empty_like(ddd))
            self.time_maa_w = nn.Parameter(torch.empty_like(ddd))
            self.time_maa_g = nn.Parameter(torch.empty_like(ddd))

            lora_rank_tokenshift = config.lora_rank_tokenshift or calc_lora_rank(0.5, 1.8)
            #lora_rank_tokenshift = 32 if n_embd < 4096 else 64

            self.time_maa_w2 = nn.Parameter(torch.empty(5, lora_rank_tokenshift, n_embd))
            self.time_maa_w1 = nn.Parameter(torch.empty(n_embd, lora_rank_tokenshift*self.time_maa_w2.size(0)))

        lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        #lora_rank_decay = 64 if n_embd < 4096 else 128

        # # per-head RWKV-6
        # H = self.num_heads
        # self.time_decay = nn.Parameter(torch.empty(H))
        # D_DECAY_LORA = 64 if n_embd < 4096 else 128
        # self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, D_DECAY_LORA))
        # self.time_decay_w2 = nn.Parameter(torch.empty(D_DECAY_LORA, H))

        # RWKV-6
        self.time_decay = nn.Parameter(torch.empty(1,1,dim_att))
        self.time_decay_w1 = nn.Parameter(torch.empty(n_embd, lora_rank_decay))
        self.time_decay_w2 = nn.Parameter(torch.empty(lora_rank_decay, dim_att))
        # self.time_faaaa = nn.Parameter(torch.empty(self.n_head, self.head_dim))

        if config.gate_rank_type == 1:
            self.gate = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        elif config.gate_rank_type == 2:
            lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)
            self.g1 = nn.Parameter(torch.empty(n_embd, lora_rank_gate))
            self.g2 = nn.Parameter(torch.empty(lora_rank_gate, self.num_heads * self.head_dim))

        if config.groupnorm_att:
            self.ln_x = nn.GroupNorm(self.num_heads, dim_att, eps=self.head_dim * 1e-5)

    def reset_parameters(self):
        print("Called reset_parameters on TMix_qwen3rwkv6 layer ", self.layer_id)

        module = self

        config = self.config

        n_layer = config.n_layer
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_id

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0

            if config.use_tokenshift:
                ddd = torch.ones(1, 1, n_embd)
                for i in range(n_embd):
                    ddd[0, 0, i] = i / n_embd

                # module.time_maa_x.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # module.time_maa_r.copy_(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                # module.time_maa_k.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                # module.time_maa_v.copy_(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                # module.time_maa_w.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))

                ddd = torch.zeros(1, 1, n_embd)
                module.time_maa_x.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                module.time_maa_r.zero_()
                module.time_maa_k.zero_()
                module.time_maa_v.zero_()
                module.time_maa_w.zero_()
                module.time_maa_g.zero_()

                module.time_maa_w2.uniform_(-0.01, 0.01)
                module.time_maa_w1.zero_()

            # # per-head RWKV-6
            # H = module.num_heads
            # # fancy time_decay
            # decay_speed = torch.ones(H)
            # for h in range(H):
            #     decay_speed[h] = -6 + 5 * (h / max(H - 1, 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            # module.time_decay.copy_(decay_speed)
            # #module.time_decay.copy_(torch.empty(H).uniform_(-8, -7))
            # module.time_decay_w1.zero_()
            # module.time_decay_w2.uniform_(-0.01, 0.01)

            # RWKV-6
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            module.time_decay.copy_(decay_speed.reshape(1,1,dim_att))
            module.time_decay_w1.zero_()
            module.time_decay_w2.uniform_(-0.01, 0.01)
            # tmp = torch.zeros(dim_att)
            # for n in range(dim_att):
            #     zigzag = ((n + 1) % 3 - 1) * 0.1
            #     tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag
            # module.time_faaaa.copy_(tmp.reshape(module.n_head, module.head_dim))

            if config.gate_rank_type == 1:
                module.gate.weight.zero_()                
            elif config.gate_rank_type == 2:
                module.g1.zero_()
                ortho_init(module.g2, 0.1)

            # FIXME - this is not in the right place to override loaded parameters
            # if config.reinit_att:
            #     module.q_proj.weight.uniform_(-0.5/((self.num_heads*self.qk_head_dim)**0.5), 0.5/((self.num_heads*self.qk_head_dim)**0.5))
            #     module.q_proj.bias.zero_()
            #     module.k_proj.weight.uniform_(-0.5/((self.num_heads*self.qk_head_dim)**0.5), 0.5/((self.num_heads*self.qk_head_dim)**0.5))
            #     module.k_proj.bias.zero_()
            #     module.v_proj.weight.uniform_(-0.5/((self.num_heads*self.head_dim)**0.5), 0.5/((self.num_heads*self.head_dim)**0.5))
            #     module.v_proj.bias.zero_()
            #     module.o_proj.weight.data.zero_()

    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        bsz, q_len, hidden_dim = x.size()

        if self.config.use_tokenshift:
            dxprev = torch.nn.functional.pad(x, (0, 0, 1, -1)) - x

            xxx = x + dxprev * self.time_maa_x
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(bsz*q_len, self.time_maa_w2.size(0), -1).transpose(0, 1)
            xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), bsz, q_len, hidden_dim)

            mr, mk, mv, mw, mg = xxx.unbind(dim=0)
            xr = x + dxprev * (self.time_maa_r + mr)
            xk = x + dxprev * (self.time_maa_k + mk)
            xv = x + dxprev * (self.time_maa_v + mv)
            xw = x + dxprev * (self.time_maa_w + mw)
            xg = x + dxprev * (self.time_maa_g + mg)
        else:
            xr = xk = xv = xw = xg = x

        query_states = self.q_proj(xr)
        key_states = self.k_proj(xk)
        value_states = self.v_proj(xv)
        decay_states = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(query_states.dtype)
        if self.config.gate_rank_type == 1:
            gate_states = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            gate_states = torch.sigmoid(xg @ self.g1) @ self.g2

        query_states = self.q_norm(query_states.view(bsz, q_len, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        decay_states = decay_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.config.use_pos_emb:
            cos, sin = shared.angles.unbind(0)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #dropout_rate = 0.0 if not self.training else self.attention_dropout

        #decay_states_log = -decay_states.float().exp()
        #decay_states_log = decay_states_log.clamp(-5) # FIXME - is this necessary?
        log_neglog_w = - 0.5 - torch.nn.functional.softplus(-decay_states.float()) # FIXME - we had tried 0-softplus before
        decay_states_log = -log_neglog_w.exp()

        if self.config.balance_state:
            key_states = (key_states * (1 - decay_states_log.exp())).to(key_states.dtype)

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

            #if 'rwkv6' in self.config.attention_type:
            #    attn_output = None # FIXME            
            #attn_output = fla_chunk_simple_gla(query_states, key_states, value_states, decay_states_log.view(bsz, self.num_heads, q_len))[0]
            #attn_output = chunk_gla(query_states, key_states, value_states, decay_states_log)[0]
            if ATTENTION_TYPE == 'gla':
                attn_output = fused_recurrent_gla(query_states, key_states, value_states, decay_states_log)[0]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            elif ATTENTION_TYPE == 'rwkv6_wind_backstepping_longhead':
                query_states,log_neglog_w,key_states,value_states = [i.transpose(1,2).to(torch.bfloat16) for i in [query_states,log_neglog_w,key_states,value_states]]
                attn_output, wkv_state = RUN_CUDA_RWKV7g(query_states, log_neglog_w, key_states, value_states) #, s0=last_state.wkv_state)
                attn_output = attn_output * key_states.shape[-1] ** -0.5
            else:
                assert False, 'bad attention type'
            if self.config.groupnorm_att:
                attn_output = self.ln_x(attn_output.view(bsz * q_len, -1)).view(bsz, q_len, -1)
            if self.config.gate_rank_type != 0:
                attn_output = attn_output * gate_states
            attn_output = self.o_proj(attn_output)
        else:
            attn_weights = (query_states * (key_states.size(-1) ** -0.5)) @ key_states.mT

            decay_states_log = decay_states_log.mean(-1, keepdim=True)
            attn_weights = attn_weights.float() * self.segsum(decay_states_log.float()) # NOTE - without the explicit cast to float ddp mismatched deepspeed here

            attn_weights = attn_weights.to(query_states.dtype)
            attn_output = torch.empty(0, device=x.device)
    
        return attn_output, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights #, past_key_value
    
class TMix_qwen3rwkv7(TMix_qwen3):
    """
    Qwen3 RWKV-7rc4a attention module, following Qwen3 attention module. This module inherits from `Qwen3Attention`
    and adds RWKV specific weights for tokenshift, decay, time_first, and the final layernorm.
    """

    def __init__(self, config:Transformer_Config, layer_id):
        super().__init__(config, layer_id)
        self.config = config

        attention_bias = True
        attention_output_bias = False

        C = self.hidden_size = config.n_embd
        N = self.head_dim = config.head_size
        self.qk_head_dim = self.head_dim
        H = self.num_heads
        attention_hidden_size = H * N
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        #self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        lora_rank_iclr = config.lora_rank_iclr or calc_lora_rank(0.5, 1.8)
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)
        lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)
        lora_rank_deformed_key = calc_lora_rank(0.5, 1.8)
        lora_rank_x = calc_lora_rank(0.8, 0.6)
        lora_rank_qk = calc_lora_rank(0.8, 0.6)

        # self.x1 = nn.Parameter(torch.empty(C, lora_rank_x))
        # self.x2 = nn.Parameter(torch.empty(lora_rank_x, C))

        # self.x_r = nn.Parameter(torch.empty(1,1,C))
        # self.x_w = nn.Parameter(torch.empty(1,1,C))
        # self.x_k = nn.Parameter(torch.empty(1,1,C))
        # self.x_v = nn.Parameter(torch.empty(1,1,C))
        # self.x_a = nn.Parameter(torch.empty(1,1,C))
        # self.x_g = nn.Parameter(torch.empty(1,1,C))

        #ratio_1_to_almost0 = 1.0 - (layer_id / self.config.n_layer)  # 1 to ~0
        #self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        # ddd = torch.zeros(1, 1, C)
        # self.time_maa_x = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_r = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_w = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_k = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_v = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_a = nn.Parameter(torch.zeros_like(ddd))
        # self.time_maa_g = nn.Parameter(torch.zeros_like(ddd))

        # self.time_maa_w2 = nn.Parameter(torch.zeros(6, lora_rank_x, C).uniform_(-0.01, 0.01))
        # self.time_maa_w1 = nn.Parameter(torch.zeros(C, lora_rank_x*self.time_maa_w2.size(0)))

        #self.q1 = nn.Parameter(torch.empty(C, lora_rank_qk))
        #self.q2 = nn.Parameter(torch.empty(lora_rank_qk, C))

        #self.k1 = nn.Parameter(torch.empty(C, lora_rank_qk))
        #self.k2 = nn.Parameter(torch.empty(lora_rank_qk, C))

        self.w0 = nn.Parameter(torch.empty(1,1,self.num_heads * self.qk_head_dim))
        self.w1 = nn.Parameter(torch.empty(C, lora_rank_decay))
        self.w2 = nn.Parameter(torch.empty(lora_rank_decay, self.num_heads * self.qk_head_dim))

        self.a0 = nn.Parameter(torch.empty(1,1,self.num_heads * self.qk_head_dim))
        self.a1 = nn.Parameter(torch.empty(C, lora_rank_iclr))
        self.a2 = nn.Parameter(torch.empty(lora_rank_iclr, self.num_heads * self.qk_head_dim))

        #if layer_id > 0:
        self.v0 = nn.Parameter(torch.empty(1,1,H*N))
        self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))
        #self.k0 = nn.Parameter(torch.empty(1,1,H*N))
        #self.k1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
        #self.k2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, H*N))

        if config.gate_rank_type == 1:
            self.gate = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        elif config.gate_rank_type == 2:
            self.g1 = nn.Parameter(torch.empty(C, lora_rank_gate))
            self.g2 = nn.Parameter(torch.empty(lora_rank_gate, self.num_heads * self.head_dim))

        #self.kk1 = nn.Parameter(torch.empty(C, lora_rank_deformed_key))
        #self.kk2 = nn.Parameter(torch.empty(lora_rank_deformed_key, C))

        self.k_k = nn.Parameter(torch.empty(1,1,self.num_heads * self.qk_head_dim))
        self.k_a = nn.Parameter(torch.empty(1,1,self.num_heads * self.qk_head_dim))
        self.r_k = nn.Parameter(torch.empty(H,N))

        #self.receptance = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        #self.key = nn.Linear(self.num_key_value_heads * self.head_dim, self.num_heads * self.qk_head_dim, bias=False)
        # self.value = nn.Linear(self.num_key_value_heads * self.head_dim, self.num_heads * self.head_dim, bias=attention_bias)
        # self.output = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_output_bias)
        if config.groupnorm_att:
            self.ln_x = nn.GroupNorm(H, H*N, eps=self.head_dim * 1e-5)

    def reset_parameters(self):
        print("Called reset_parameters on TMix_qwen3rwkv7 layer ", self.layer_id)

        module = self

        num_hidden_layers = n_layer = self.config.n_layer
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_id

        ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

        # time_weight = torch.tensor(
        #     [i / C for i in range(C)],
        #     dtype=module.x_k.dtype,
        #     device=module.x_k.device,
        # )
        # time_weight = time_weight[None, None, :]

        decay_speed = [
            -7.0 + 5.0 * (n / (self.num_heads * self.qk_head_dim - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            for n in range(self.num_heads * self.qk_head_dim)
        ]

        # def inverse_sigmoid(x): return math.log(x) - math.log(1 - x)
        # decay_speed = [
        #     inverse_sigmoid(0.995) + (inverse_sigmoid(0.875)-inverse_sigmoid(0.995)) * (n / (attention_hidden_size - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
        #     for n in range(attention_hidden_size)
        # ]
        decay_speed = torch.tensor(decay_speed, dtype=module.w0.dtype, device=module.w0.device)

        with torch.no_grad():
            # torch.nn.init.zeros_(module.x_r) #.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )
            # torch.nn.init.zeros_(module.x_w) #.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
            # torch.nn.init.zeros_(module.x_k) #.copy_( 1.0 - (torch.pow(time_weight, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1) )
            # torch.nn.init.zeros_(module.x_v) #.copy_( 1.0 - (torch.pow(time_weight, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1) )
            # torch.nn.init.zeros_(module.x_a) #.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
            # torch.nn.init.zeros_(module.x_g) #.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )
            
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            #module.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))

            # module.x1.zero_()
            # ortho_init(module.x2, 0.1)

            # module.q1.zero_()
            # ortho_init(module.q2, 0.1)
            # module.k1.zero_()
            # ortho_init(module.k2, 0.1)

            module.w0.copy_(decay_speed.reshape(1,1,-1) + 0.5) # !!! 0.5 comes from F.softplus !!!
            #module.w0.copy_(decay_speed.reshape(1,1,-1) - 1.0)
            #module.w0.copy_(-2.0 + 1e-5 * torch.randn(1, 1, dim_att))
            module.w1.zero_()
            ortho_init(module.w2, 0.1)

            module.a0.zero_()
            module.a1.zero_()
            ortho_init(module.a2, 0.1)

            if layer_id > 0:
                module.v0.copy_(1.0)
                module.v1.zero_()
                ortho_init(module.v2, 0.1)
                #module.k0.copy_(1.0)
                #module.k1.zero_()
                #ortho_init(module.k2, 0.1)

            if self.config.gate_rank_type == 1:
                module.gate.weight.zero_()
            elif self.config.gate_rank_type == 2:
                module.g1.zero_()
                ortho_init(module.g2, 0.1)

            #module.kk1.zero_()
            #ortho_init(module.kk2, 0.1)
            
            module.k_k.copy_(0.85) # FIXME - should this be 1.0?
            module.k_a.copy_(1.0)
            module.r_k.zero_()

            #module.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/((self.num_heads*self.qk_head_dim)**0.5))
            #module.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/((self.num_heads*self.qk_head_dim)**0.5))
            #module.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(attention_hidden_size**0.5))
            # module.output.weight.data.zero_()
            #module.key.weight.data.uniform_(-0.05/(self.num_key_value_heads**0.5), 0.05/(attention_hidden_size**0.5))
            #module.value.weight.data.uniform_(-0.05/(self.num_key_value_heads**0.5), 0.05/(attention_hidden_size**0.5))
            # ortho_init(module.key.weight, 0.1)
            # ortho_init(module.value.weight, 0.1)
            # module.key.weight.data.zero_()
            # module.value.weight.data.zero_()


    def forward(self, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state:ModelState, shared:Shared, output_attentions:bool=False):
        last_state = last_model_state.block_states[self.layer_id].time_mix_state
        # bsz, q_len, hidden_dim = x.size()
        # B, L, D = x.size()
        # QH = self.num_heads
        # KVH = self.num_key_value_heads


        input_seq_len = x.size(1)
        if self.training and 'rwkv7_fla' not in self.config.attention_type and input_seq_len % 16 != 0:
            x = F.pad(x, (0, 0, 0, 16 - input_seq_len%16))
        B, T, C = x.size()
        H = self.num_heads
        N = self.head_dim

        # dxprev = F.pad(x, (0, 0, 1, -1)) - x

        # xxx = x + dxprev * self.time_maa_x
        # xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 6, -1).transpose(0, 1)
        # xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, -1)
        # mr, mw, mk, mv, ma, mg = xxx.unbind(dim=0)

        # xr = x + dxprev * (self.time_maa_r + mr)
        # xw = x + dxprev * (self.time_maa_w + mw)
        # xk = x + dxprev * (self.time_maa_k + mk)
        # xv = x + dxprev * (self.time_maa_v + mv)
        # xa = x + dxprev * (self.time_maa_a + ma)
        # xg = x + dxprev * (self.time_maa_g + mg)

        # xr = x+dxprev*self.x_r
        # xw = x+dxprev*self.x_w
        # xk = x+dxprev*self.x_k
        # xv = x+dxprev*self.x_v
        # xa = x+dxprev*self.x_a
        # xg = x+dxprev*self.x_g

        #x = x + F.tanh(x @ self.x1) @ self.x2
        xr = xw = xk = xv = xa = xg = x

        r = self.q_norm(self.q_proj(xr).view(B,T,-1,N)).view(B,T,-1)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.k_norm(self.k_proj(xk).view(B,T,-1,N)).view(B,T,-1)
        v = self.v_proj(xv)
        # dk = self.key(torch.tanh(k))
        # dv = self.value(torch.tanh(v))
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        if self.config.gate_rank_type == 1:
            g = torch.sigmoid(self.gate(xg))
        elif self.config.gate_rank_type == 2:
            g = torch.sigmoid(xg @ self.g1) @ self.g2
        

        # FIXME - adding w0 twice here!!!
        log_neglog_w = - 0.5 - torch.nn.functional.softplus(-(self.w0 + w).float()) # FIXME - we had tried 0-softplus before

        if self.config.use_pos_emb:
            r = r.view(B,T,-1,N)
            k = k.view(B,T,-1,N)
            r = r.transpose(1,2) # BHTN
            k = k.transpose(1,2) # B(kvh)TN
            cos, sin = shared.angles.unbind(0)
            r, k = apply_rotary_pos_emb(r, k, cos, sin)
            r = r.transpose(1,2).view(B,T,-1).to(v.dtype)
            k = k.transpose(1,2).view(B,T,-1).to(v.dtype)

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        v = v.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)

        # k = k.view(B,T,H,-1).float()
        # k = (k / (torch.norm(k, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)
        # kk = k
        #kk = torch.nn.functional.normalize((k * self.k_k).view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        #kk = (k * self.k_k).view(B,T,H,-1).float()
        kk = (k).view(B,T,H,-1).float()
        kk = (kk / (torch.norm(kk, dim=-1, keepdim=True) + 1e-12)).view(B,T,-1).to(k.dtype)
        #k = k * (1 + (a-1) * self.k_a)

        if self.layer_id == 0:
            v_first = v
            #v_first = torch.cat([k, v], dim=-1)
        else:
            #k_kv_first, v_kv_first = torch.chunk(kv_first, chunks=2, dim=-1)
            #k = k + (k_kv_first - k) * torch.sigmoid(self.k0 + (xk @ self.k1) @ self.k2)
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        z = -kk
        b = kk*a
        #z = b = torch.zeros_like(k)
        w = (-log_neglog_w.exp()).exp()
        # s'^T = ws^T-as^Tkk^T+avk^T, and I assume k is normalized
        # the left part removes 1-w 'values'  via multiplication
        # the middle part removes a 'values'
        # the right part adds back in a new 'values'
        # so the total amount removed is 1-w+a 'values', but the amount added is a
        # to compensate, we need to somehow adjust a on the right side to become 1-w+a
        # leaving us with the revised recurrence formula:
        # s'^T = ws^T-as^Tkk^T+(1-w+a)vk^T, and I assume k is normalized
        if self.config.balance_state:
            k = k * (1-w+a)
        r,log_neglog_w,k,v,z,b = [i.to(torch.bfloat16).view(B,T,H,-1) for i in [r,log_neglog_w,k,v,z,b]]
        x, wkv_state = RUN_CUDA_RWKV7g(r, log_neglog_w, k, v, z, b) #, s0=last_state.wkv_state)

        xdtype = x.dtype
        if self.config.groupnorm_att:
            x = F.group_norm(x.view(B * T, -1).float(), self.ln_x.num_groups, self.ln_x.weight.float(), self.ln_x.bias.float(), self.ln_x.eps)
        else:
            x = x * N ** -0.5
        x = x.view(B, T, -1).to(xdtype)
        if self.config.gate_rank_type != 0:
            x = x * g
        x = self.o_proj(x)

        if input_seq_len != T:
            x = x[:, :input_seq_len]

        attn_weights = torch.empty(0, device=x.device)
        return x, v_first, TimeMixState(wkv_state, last_state.shift_state), attn_weights #, past_key_value

        #return x, None, past_key_value

def get_cmix_default_state(x:Tensor, config:Transformer_Config, requires_grad:bool):
    B, T, C = x.size()
    return ChannelMixState(
        torch.zeros([B, C], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
    )

class CMix_qwen3(nn.Module):
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

def is_layer_attention(config, layer_id):
    return layer_id >= config.n_layer - config.preserve_last_n_layers and (layer_id > min(config.n_layer, config.last_striping_layer) or (min(config.n_layer-1, config.last_striping_layer) - layer_id) % config.attention_striping <= 1)

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config:TrainerCLI_Config, layer_id:int):
        super().__init__()

        self.config = config
        self.layer_id = layer_id

        args:Transformer_Config = config.model

        self.input_layernorm = Qwen3RMSNorm(args.n_embd, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(args.n_embd, eps=args.rms_norm_eps)

        cmix = CMix_qwen3(args, layer_id)

        if is_layer_attention(args, layer_id):
            attn_class = TMix_qwen3newatt # TMix_qwen3doubleatt
        elif 'rwkv6' in args.attention_type or 'gla' in args.attention_type:
            attn_class = TMix_qwen3rwkv6
        elif 'rwkv7' in args.attention_type:
            attn_class = TMix_qwen3rwkv7
        elif 'softpick' in args.attention_type:
            attn_class = TMix_qwen3softpick
        else:
            attn_class = TMix_qwen3
        self.self_attn = attn_class(args, layer_id)
        self.default_time_mix_state_factory = self.self_attn.get_default_state_factory() if hasattr(self.self_attn, 'get_default_state_factory') else lambda x, c, r: TimeMixState()

        self.teacher_attn = None
        if config.train is not None:
            if config.train.attention_distillation_stage in (0, 11, 1):
                self.teacher_attn = TMix_qwen3(args, layer_id)
        
        self.default_channel_mix_state_factory = cmix.get_default_state_factory() if hasattr(cmix, 'get_default_state_factory') else lambda x, c, r: ChannelMixState()
        self.mlp = cmix

    def forward(self, x:Tensor, residual_first:Tensor, frozen_residual:Tensor, reset_mask:Tensor, v_first:Tensor, last_model_state:ModelState, shared:Shared, output_attentions:bool, output_post_attention_hidden_states:bool):
        #residual_first = self.input_layernorm(residual_first)
		#frozen_residual = self.input_layernorm(frozen_residual)

        s = last_model_state
        if self.teacher_attn is not None:
            dx, _, last_timemix_state, attentions = self.teacher_attn(self.input_layernorm(x), residual_first, frozen_residual, reset_mask, v_first, s, shared, output_attentions)
            student_dx, v_first, student_last_timemix_state, student_attentions = self.self_attn(self.input_layernorm(x), residual_first, self.input_layernorm(frozen_residual), reset_mask, v_first, s, shared, output_attentions)
            s.block_states[self.layer_id].time_mix_state = student_last_timemix_state
        else:
            dx, v_first, last_timemix_state, attentions = self.self_attn(self.input_layernorm(x), residual_first, frozen_residual, reset_mask, v_first, s, shared, output_attentions)
            student_dx, student_last_timemix_state, student_attentions = None, None, None
            s.block_states[self.layer_id].time_mix_state = last_timemix_state

        if output_post_attention_hidden_states:            
            post_attention_hidden_states = dx
            student_post_attention_hidden_states = student_dx
            # # FIXME - we change the meaning of post_attention_hidden_states to mean the difference here
            # post_attention_hidden_states = torch.linalg.vector_norm(dx - student_dx, dim=-1).mean() * (dx[0].size(-1) ** -0.5)
            # student_post_attention_hidden_states = torch.empty(0, device=x.device)
        else:
            post_attention_hidden_states = torch.empty(0, device=x.device)
            student_post_attention_hidden_states = torch.empty(0, device=x.device)
            
        if dx.size(1) != x.size(1):
            #print("ADJUSTING SIZE OF RESIDUAL FROM ", x.size(1), " TO ", dx.size(1))
            if dx.size(1) > x.size(1):
                assert dx.size(1) == x.size(1) * 2
                x = torch.cat([x, x], dim=1)
            else:
                x = F.pad(x, [0, 0, dx.size(1) - x.size(1), 0])
        x = x + dx
        dx, last_chanmix_state = self.mlp(self.post_attention_layernorm(x), s)
        x = x + dx
        return x, v_first, s, attentions, post_attention_hidden_states, student_attentions, student_post_attention_hidden_states

def ckpt(block:Qwen3DecoderLayer, *block_args):
    if block.training and block.config.train.grad_cp == 1: # and 'fsdp' not in block.config.train.strategy: # FSDP has its own checkpointing wrapper
        #if "deepspeed" in block.config.train.strategy:
        #    results = deepspeed.checkpointing.checkpoint(block, *block_args)
        #else:
        # NOTE - both deepspeed.checkpointing.checkpoint and use_reentrant=True failed miserably (bad loss) when used in conjunction with requires_grad=False params and grad_cp with deepspeed
        results = torch.utils.checkpoint.checkpoint(block, *block_args, use_reentrant=False)
    else:
        results = block(*block_args)
    return results

class Qwen3Decoder(nn.Module):
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        self.shared = Shared()

        self.embed_tokens = nn.Embedding(args.vocab_size, args.n_embd, args.vocab_padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(args.n_layer)]
        )
        self.norm = Qwen3RMSNorm(args.n_embd, eps=args.rms_norm_eps)

    def forward_preamble(self, x, last_model_state:ModelState|None = None, ):
        config : Transformer_Config = self.config.model

        B, T, C = x.size()

        shared = self.shared
        if config.rope is not None and T > shared.angles.size(0):
            max_ctx_len = max(config.ctx_len, (T + 15) // 16 * 16)
            shared.angles = generate_rotary_embedding(max_ctx_len, config.head_size, config.rope.base * config.rope.rebase, config.rope.rescale).to(self.norm.weight)

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

        eos_token_id = 151643 # "<|endoftext|>"
        reset_mask = token_ids == eos_token_id # state needs to be reset at the beginning of the EOS token processing, since we're predicting the next token based on ZERO state!

        x = self.embed_tokens(token_ids)
        residual_first = frozen_residual = x

        v_first = torch.tensor([], device=x.device)
        last_model_state = self.forward_preamble(x, last_model_state)

        hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs = (), (), ()
        student_hidden_states_outputs, student_attentions_outputs, student_post_attention_hidden_states_outputs = (), (), ()
        if output_hidden_states:
            hidden_states_outputs += (x,)
            student_hidden_states_outputs += (x,)
        for decoder_layer in self.layers:
            # if decoder_layer.layer_id <= config.n_layer - config.preserve_last_n_layers:
            #     #frozen_residual = x
            #     frozen_residual = rms_norm(x)
            #     #freezy = x
            #     #frozen_residual = decoder_layer.input_layernorm(x)
            #     #frozen_residual = rms_norm(self.decompress(torch.cat([residual_first, x], dim=-1)))
            # #frozen_residual = decoder_layer.input_layernorm(freezy)
            x, v_first, last_model_state, attentions, post_attention_hidden_states, student_attentions, student_post_attention_hidden_states = ckpt(decoder_layer, x, residual_first, frozen_residual, reset_mask, v_first, last_model_state, self.shared, output_attentions, output_post_attention_hidden_states)
            hidden_states_outputs += (x,)
            student_hidden_states_outputs += (x,)
            if output_attentions:
                attentions_outputs += (attentions,)
                student_attentions_outputs += (student_attentions,)
            if output_post_attention_hidden_states:
                post_attention_hidden_states_outputs += (post_attention_hidden_states,)
                student_post_attention_hidden_states_outputs += (student_post_attention_hidden_states,)

        x = self.norm(x)

        # FIXME - we added one after the norm so we can include that (calling it from externally messes up FSDP)
        if output_hidden_states:
            hidden_states_outputs += (x,)
            student_hidden_states_outputs += (x,)

        return LLMOutput(x, last_model_state, hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs, student_attentions_outputs, student_post_attention_hidden_states_outputs)
        #return x, last_model_state, hidden_states_outputs, attentions_outputs, post_attention_hidden_states_outputs # FIXME - not updating state at all

class Model_qwen3(nn.Module): # Qwen3CausalLM
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()

        self.config = config

        args:Transformer_Config = config.model

        if args.dim_att <= 0:
            args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

        self.model = None

    def configure_model(self):
        if self.model is not None: 
            return

        self.model = Qwen3Decoder(self.config)

        self.lm_head = nn.Linear(self.config.model.n_embd, self.config.model.vocab_size, bias=False)

    def set_grads(self):
        train_config = self.config.train
        if train_config is not None:
            if train_config.attention_distillation_stage in (0, 11, 1):
                self.requires_grad_(False)
                for decoder_layer in self.model.layers:
                    if train_config.attention_distillation_stage == 0:
                        decoder_layer.self_attn.time_maa_x.requires_grad_(True)
                        decoder_layer.self_attn.time_maa_r.requires_grad_(True)
                        decoder_layer.self_attn.time_maa_k.requires_grad_(True)
                        decoder_layer.self_attn.time_maa_w.requires_grad_(True)
                        decoder_layer.self_attn.time_maa_w1.requires_grad_(True)
                        decoder_layer.self_attn.time_maa_w2.requires_grad_(True)
                        decoder_layer.self_attn.time_decay.requires_grad_(True)
                        decoder_layer.self_attn.time_decay_w1.requires_grad_(True)
                        decoder_layer.self_attn.time_decay_w2.requires_grad_(True)
                        # FIXME - wow we removed q, k here by accident and it.. helped??!?!
                        # decoder_layer.self_attn.q_proj.requires_grad_(True)
                        # decoder_layer.self_attn.k_proj.requires_grad_(True)
                    elif train_config.attention_distillation_stage == 11:
                        decoder_layer.self_attn.requires_grad_(True)
                        #decoder_layer.self_attn.q_proj.requires_grad_(False)
                        #decoder_layer.self_attn.k_proj.requires_grad_(False)
                        # decoder_layer.self_attn.v_proj.requires_grad_(False)
                        # decoder_layer.self_attn.o_proj.requires_grad_(False)
                    elif train_config.attention_distillation_stage == 1:
                        # if decoder_layer.layer_id < self.config.model.n_layer - self.config.model.preserve_last_n_layers:
                        decoder_layer.self_attn.requires_grad_(True)
                        # decoder_layer.self_attn.k_proj.requires_grad_(False)
                        # decoder_layer.self_attn.v_proj.requires_grad_(False)
            elif train_config.attention_distillation_stage == 21:
                self.requires_grad_(False)
                for decoder_layer in self.model.layers:
                    decoder_layer.self_attn.requires_grad_(True)
            elif train_config.attention_distillation_stage == 31:
                self.requires_grad_(False)
                for decoder_layer in self.model.layers:
                    decoder_layer.self_attn.time_maa_w.requires_grad_(True)
                    decoder_layer.self_attn.time_maa_w1.requires_grad_(True)
                    decoder_layer.self_attn.time_maa_w2.requires_grad_(True)
                    decoder_layer.self_attn.time_decay.requires_grad_(True)
                    decoder_layer.self_attn.time_decay_w1.requires_grad_(True)
                    decoder_layer.self_attn.time_decay_w2.requires_grad_(True)
            # elif train_config.attention_distillation_stage == 2:
            #     layer_id = -1
            #     for decoder_layer in self.model.layers:
            #         layer_id += 1
            #         if layer_id >= self.config.model.n_layer - self.config.model.preserve_last_n_layers:
            #             decoder_layer.requires_grad_(False)

            # elif train_config.attention_distillation_stage == 2:
            #     # Harrison's possible fix for v_first explosion during stage 3
            #     for decoder_layer in self.model.layers:
            #         decoder_layer.self_attn.v_proj.requires_grad_(False)
            #         break
            # elif train_config.attention_distillation_stage == 2:
            #     self.requires_grad_(False)
            #     # self.model.embed_tokens.requires_grad_(False)
            #     # self.model.norm.requires_grad_(False)
            #     # self.lm_head.requires_grad_(False)
            #     layer_id = -1
            #     for decoder_layer in self.model.layers:
            #         layer_id += 1
            #         if layer_id >= self.config.model.n_layer // 2:
            #             decoder_layer.requires_grad_(True)

            #     self.requires_grad_(False)
            #     # self.model.embed_tokens.requires_grad_(False)
            #     # self.model.norm.requires_grad_(False)
            #     # self.lm_head.requires_grad_(False)

            #         decoder_layer.self_attn.requires_grad_(True)
            #         # decoder_layer.self_attn.k_proj.requires_grad_(False)
            #         # decoder_layer.self_attn.v_proj.requires_grad_(False)

            # FIXME - remove these for full training
            # for decoder_layer in self.model.layers:
            #     decoder_layer.post_attention_layernorm.requires_grad_(False)
            #     decoder_layer.mlp.requires_grad_(False)
            # self.model.embed_tokens.requires_grad_(False)
            # self.model.norm.requires_grad_(False)
            # self.lm_head.requires_grad_(False)

    def forward(self, token_ids:Tensor|list, last_model_state:ModelState|None = None, output_hidden_states:bool=False, output_attentions:bool=False, output_post_attention_hidden_states:bool=False):
        #print("teacher q min, max", float(self.model.layers[0].self_attn.q_proj.weight.min()), float(self.model.layers[0].self_attn.q_proj.weight.max()))
        results = self.model(token_ids, last_model_state=last_model_state, output_hidden_states=output_hidden_states, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)
        results.logits = self.lm_head(results.logits)
        return results
    
    def get_optim_groups(self):
        self.set_grads()

        train_config = self.config.train

        # separates groups for weight decay and non-weight decay

        # JIT at last minute
        for decoder_layer in self.model.layers:
            decoder_layer.self_attn = TJIT(decoder_layer.self_attn)
            decoder_layer.mlp = TJIT(decoder_layer.mlp)

        lr_decay = set()
        lr_1x = set()
        lr_fp32 = set()
        lr_2 = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            #if '.mlp.' in n and int(n.split('.')[2]) >= self.config.model.n_layer - self.config.model.preserve_last_n_layers:
            #    continue
            # if 'lm_head' in n or 'embed_tokens' in n:
            #     lr_fp32.add(n)
            #     continue
            if '.self_attn.' in n and not is_layer_attention(self.config.model, int(n.split('.')[2])):
                lr_2.add(n)
                continue
            # FIXME - this seems to do nothing, but it was what we used when training the good result
            # if '.mlp.' in n and int(n.split('.')[2]) >= self.config.model.n_layer - self.config.model.preserve_last_n_layers:
            #     lr_1x.add(n)
            #     continue
            # NOTE - this check is no good in FSDP because the tensors end up with some stupid fake shape
            #if (len(p.squeeze().shape) >= 2) and (train_config.weight_decay > 0):
            if train_config.weight_decay > 0 and '.bias' not in n and 'norm' not in n and 'ln' not in n:
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        param_dict = {n: p for n, p in self.named_parameters()}
        #param_check = list(lr_decay) + list(lr_1x) + list(lr_fp32)
        #if not train_config.load_partial and (train_config.teacher is None or train_config.teacher.attention_distillation_stage == 2):
        #    assert sorted(param_dict) == sorted(param_check)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_fp32 = sorted(list(lr_fp32))
        lr_2 = sorted(list(lr_2))
        
        print('1x', len(lr_1x), '\n')
        print('decay', len(lr_decay), '\n')
        print('fp32', len(lr_fp32), '\n')
        print('high', len(lr_2), '\n')
        
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0, 'name':'lr_1x'},
        ]
        if len(lr_fp32) > 0:
            # FIXME - NOTE THIS VERY LOW LR SCALE!!!
            optim_groups += [{"params": [param_dict[n] for n in lr_fp32], "weight_decay": train_config.weight_decay, "my_lr_scale": 0.5, 'name':'lr_fp32'}]
        if len(lr_2) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_2], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_2'}]
        if len(lr_decay) > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_decay'}]

        return optim_groups    

    # def _init_weights(self, module):
    #     std = 0.02 #self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

    def init_all_weights(self):
        # FIXME - we really need to init the gate here to identity (zero weights, ones bias) but instead we just won't init weights since they're all grabbed or set anyway
        pass
        # self.apply(self._init_weights)
        # for n, p in self.named_parameters():
        #     requires_grad_temp = p.requires_grad
        #     p.requires_grad_(False)
        #     if n.endswith('.ln_x.weight'):
        #         layer_scale = (1+int(n.split('.')[2])) / self.config.model.n_layer
        #         print('.ln_x.weight layer', int(n.split('.')[2]), "scale", (layer_scale ** 0.7))
        #         p *= 0.0
        #         p += (layer_scale ** 0.7)
        #     p.requires_grad = requires_grad_temp
