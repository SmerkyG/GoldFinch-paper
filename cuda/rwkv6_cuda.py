import os
import torch
from torch import nn, Tensor
from src.CoreDependencies import *

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MODEL_TYPE"]:
    extra_cuda_cflags = ["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}",]
    if torch.cuda.is_available():
        if torch.version.hip:
            extra_cuda_cflags += ["--save-temps"]
        else:
            extra_cuda_cflags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

    load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=extra_cuda_cflags)
        
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, k, v, w, u, s):
            with torch.no_grad():
                dtype = r.dtype
                assert r.dtype == dtype, 'mismatched r.dtype'
                assert k.dtype == dtype, 'mismatched k.dtype'
                assert v.dtype == dtype, 'mismatched v.dtype'
                assert w.dtype == dtype, 'mismatched w.dtype'
                assert u.dtype == dtype, 'mismatched u.dtype'
                assert s.dtype == dtype, 'mismatched s.dtype'
                B, T, C = r.size()
                H = C // HEAD_SIZE
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                ctx.dtype = dtype
                assert r.is_contiguous(), 'r not contiguous'
                assert k.is_contiguous(), 'k not contiguous'
                assert v.is_contiguous(), 'v not contiguous'
                assert w.is_contiguous(), 'w not contiguous'
                assert u.is_contiguous(), 'u not contiguous'
                assert s.is_contiguous(), 's not contiguous'
                ctx.save_for_backward(r, k, v, w, u)
                y = torch.empty((B, T, C), device=r.device, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                if dtype == torch.bfloat16:
                    torch.ops.wkv6b.forward_bf16(B, T, C, H, r, k, v, w, u, s, y)
                elif dtype == torch.float16:
                    torch.ops.wkv6b.forward_fp16(B, T, C, H, r, k, v, w, u, s, y)
                elif dtype == torch.float32:
                    torch.ops.wkv6b.forward_fp32(B, T, C, H, r, k, v, w, u, s, y)
                else:
                    raise ValueError(f"Unsupported dtype {dtype} for WKV_6")
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                dtype = ctx.dtype
                assert gy.dtype == dtype, 'mismatched gy.dtype'
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                if not gy.is_contiguous():
                    gy = gy.contiguous()
                assert gy.is_contiguous(), 'gy not contiguous'
                r, k, v, w, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                #gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                if dtype == torch.bfloat16:
                    torch.ops.wkv6b.backward_bf16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                elif dtype == torch.float16:
                    torch.ops.wkv6b.backward_fp16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                elif dtype == torch.float32:
                    torch.ops.wkv6b.backward_fp32(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                else:
                    raise ValueError(f"Unsupported dtype {dtype} for WKV6_CUDA")
                gu = torch.sum(gu, 0).view(H, C//H)
                return (gr, gk, gv, gw, gu, None)

    @TCompileDisable 
    def RUN_CUDA_RWKV6(r, k, v, w, u, s) -> Tensor:
        return WKV_6.apply(r, k, v, w, u, s)
else:
    @TCompileDisable 
    def RUN_CUDA_RWKV6(r, k, v, w, u, s) -> Tensor:
        return None    
