# This file is modified from https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py and is separately licensed according to the following license:
"""
MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

def apply_chunked(fn, chunk_size, *args):
    n = args[0].numel()
    for begin in range(0, n, chunk_size):
        fn(*[arg.view(n)[begin:begin+chunk_size] if isinstance(arg, torch.Tensor) and arg.numel() > 1 else arg for arg in args])

# update functions

def update_fn(p, grad, scaled_exp_avg, fp16_multiplier, lr, wd, beta1, beta2):
    # stepweight decay

    p.mul_(1 - lr * wd)

    # weight update

    grad = grad * fp16_multiplier

    update = scaled_exp_avg.clone().float().mul_(beta1).add_(grad, alpha = 1 - beta1).sign_()
    p.add_(update, alpha = -lr)

    # decay the momentum running average coefficient

    new_exp_avg = scaled_exp_avg.clone().float().mul_(beta2).add_(grad, alpha = 1 - beta2)
    scaled_exp_avg.copy_(new_exp_avg)


# class

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
        use_fp16: bool = False,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay,
            use_fp16 = use_fp16
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, use_fp16, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], group['use_fp16'], self.state[p]

                # init state - exponential moving average of gradient values

                fp16_multiplier = 32768.0 if use_fp16 else 1.0

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros(p.shape, dtype=torch.float16 if use_fp16 else p.dtype, device=p.device)

                scaled_exp_avg = state['exp_avg']

                apply_chunked(
                    self.update_fn,
                    64*1024*1024,
                    p,
                    grad,
                    scaled_exp_avg,
                    fp16_multiplier,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss