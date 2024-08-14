import os, math, gc, importlib
import torch
import torch.utils.checkpoint
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

from .state import ModelState

import src.metrics as metrics

from src.logger import print0 as print

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model:nn.Module, config:TrainerCLI_Config):
        super().__init__()
        self.model = model
        self.config = config
        self.metrics = dict(loss=metrics.Loss(), acc=metrics.Accuracy())

    def forward(self, idx, last_model_state:ModelState|None = None):
        return self.model.forward(idx, last_model_state)
    
    def configure_optimizers(self):
        train_config = self.config.train
        
        optim_groups = self.model.get_optim_groups()

        betas = (train_config.beta1, train_config.beta2)
        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return FusedAdam(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False


    def _get_loss_logits_preds(self, batch, batch_idx, last_model_state):
        x, y = batch
        logits, next_model_state = self(x, last_model_state)
    
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.flatten())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

        if loss.isinf().any():
            raise Exception("loss was infinite")

        if loss.isnan().any():
            raise Exception("loss was NaN")

        return loss, logits, preds, next_model_state
    
    def get_real_global_step(self): return int(self.trainer.global_step)
    def get_real_tokens(self): return self.get_real_global_step() * self.config.model.ctx_len * self.config.runtime.global_step_bsz
    def get_real_progress(self):
        config = self.config
        progress = self.get_real_tokens() / abs(config.train.my_exit_tokens)
        progress = max(0, min(1, progress))
        return progress
    def get_lr_progress(self):
        config = self.config
        wait_tokens = int(config.train.lr_wait * abs(config.train.my_exit_tokens))
        warmup_tokens = config.train.warmup_steps * config.model.ctx_len * config.runtime.global_step_bsz
        token_offset = warmup_tokens + wait_tokens
        progress = (self.get_real_tokens() - token_offset) / (abs(config.train.my_exit_tokens) - token_offset)
        progress = max(0, min(1, progress))
        return progress


    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        model_state = None

        loss, logits, preds, model_state = self._get_loss_logits_preds((inputs, labels), batch_idx, model_state)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        # FIXME - sync from other devices/nodes here
        for metric in self.metrics.values():
            metric.update(margs)
        if self.trainer.is_global_zero:
            self.log("loss", float(loss), prog_bar=True, on_step=True)#, rank_zero_only=True)
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                    logdict = dict(tokens = self.get_real_tokens())
                    #str = f"epoch:{self.current_epoch} token:{self.all_nodes_tokens_processed:,} step:{batch_idx} "
                    for name, metric in self.metrics.items():
                        metric_value = metric.compute()
                        logdict['train/' + name] = metric_value
                        metric.clear()
                        #str += f'{name}:{metric_value:.4f} '
                    #str += f"{gb:.1f}gb {int(ms_per)}ms {ktok_per_sec:.2f}kT/s {self.total_runtime:.1f}sec"
                    #print(str)
                    if len(self.config.train.wandb) > 0:
                        self.trainer.my_wandb.log(logdict, step=self.get_real_global_step(), commit=True)

        return L2Wrap.apply(loss, logits)

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"STARTING VALIDATION")
            print()

            # clear metrics
            for metric in self.metrics.values():
                metric.compute()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            logdict = dict(tokens = self.get_real_tokens())
            str = f"VALIDATION COMPLETE. "
            for name, metric in self.metrics.items():
                metric_value = metric.compute()
                logdict["val/" + name] = metric_value
                str += f"{metric_value:.4f} "
                metric.clear()
            if len(self.config.train.wandb) > 0:
                self.trainer.my_wandb.log(logdict, step=self.get_real_global_step(), commit=True)

            console_clear_last_line()
            print(str)
            print()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, logits, preds, next_block_states = self._get_loss_logits_preds(batch, batch_idx, None)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for name, metric in self.metrics.items():
            metric.update(margs)
            # on_epoch causes this to be logged in aggregate rather than per batch
            #self.log('val/'+name, metric.compute(), on_epoch=True, rank_zero_only=True)
            #metric.clear()
        #self.log("tokens", float(self.all_nodes_tokens_processed), on_epoch=True, rank_zero_only=True)
        return loss

    # def training_step_end(self, batch_parts):
    #     if pl.__version__[0]!='2':
    #         all = self.all_gather(batch_parts)
    #         if self.trainer.is_global_zero:
    #             self.trainer.my_loss_all = all