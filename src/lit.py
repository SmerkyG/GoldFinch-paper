import os, math, gc, importlib
import torch
import torch.linalg
import torch.utils.checkpoint
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import pickle
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader    

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig

from .state import ModelState

import src.metrics as metrics

from src.logger import print0 as print

if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    from deepspeed.utils import safe_set_full_fp32_param, safe_set_local_fp32_param, safe_set_full_optimizer_state

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
    def __init__(self, model:nn.Module, config:TrainerCLI_Config, teacher:nn.Module|None=None):
        super().__init__()
        self.model = model
        self.config = config
        self.teacher = teacher
        self.metrics = dict(loss=metrics.Loss(), acc=metrics.Accuracy())
        self.configured = False

    def configure_model(self):
        if self.configured:
            return
        self.configured = True

        if hasattr(self.model, 'configure_model'):
            self.model.configure_model()

        if self.config.train is not None:
            if self.config.train.load_model == '' or (self.config.train.load_partial and self.config.train.attention_distillation_stage != 3):
                self.init_all_weights()

        if 'deepspeed_stage_3' not in self.config.train.strategy:
            self.load_weights()

    def init_all_weights(self):
        print("Initializing weights...")
        if hasattr(self.model, 'init_all_weights'):
            self.model.init_all_weights()

    def configure_gradient_clipping(
            self,
            optimizer,
            gradient_clip_val = None,
            gradient_clip_algorithm = None,
    ):
        if 'fsdp' in self.config.train.strategy:
            assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
            #self.model.clip_grad_norm_(gradient_clip_val)
            #self.clip_grad_by_norm(optimizer, clip_val)
            #self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
            #if gradient_clip_algorithm == 'norm':
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
            self.trainer.strategy.model.clip_grad_norm_(gradient_clip_val) #self.config.train.gradient_clip_val)
        else:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)


    def save_weights(self, path):
        print("saving ", path)
        config = self.config

        model = self.model
        if 'fsdp' in config.train.strategy:
            # NOTE - this is how we get the FSDP wrapped model - if you use self you won't get the right output saved!!!
            model:nn.Module = self.trainer.strategy.model
            # annoyingly, we are REQUIRED to get the state dict from the FSDP module, which is only the top level LightningModelWrapper
            # so, get it, then edit the dict to remove the `model.` prefix

            assert(any(isinstance(m, FSDP) for m in model.modules()))
            # FIXME - context manager was crashing on release
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
            # with FSDP.state_dict_type(
            #     model,
            #     StateDictType.FULL_STATE_DICT,
            #     FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            # ):
            save_dict = model.state_dict()
            for k in list(save_dict.keys()):
                if k.startswith('model.'):
                    save_dict[k[len('model.'):]] = save_dict[k]
                    del save_dict[k]
        elif 'deepspeed_stage_3' not in config.train.strategy:
            save_dict = model.state_dict()
        else:
            
            # FIXME - this would save the whole model as well as optimizer state and dataset state
            #self.trainer.save_checkpoint(path, weights_only=True,)

            def save(module: torch.nn.Module, prefix: str = "", save_dict:dict|None=None) -> dict:
                if save_dict is None:
                    save_dict = {}
                #print("saving prefix", prefix)
                with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        for n, p in module.named_parameters():
                            save_dict[prefix + n] = p.detach().cpu()

                for name, child in module._modules.items():
                    if child is not None:
                        save(child, prefix + name + ".", save_dict)
                        
                return save_dict

            save_dict = save(model)

        if self.trainer.local_rank == 0:
            torch.save(save_dict, path)

    def load_weights(self):
        config = self.config
        ckpt_path = config.train.load_model
        if ckpt_path != '':
            self.load_model_weights(self.model, ckpt_path)
        
        if self.teacher is not None and config.train.teacher is not None:
           teacher_ckpt_path = config.train.teacher.path
           if teacher_ckpt_path != '':
               self.load_model_weights(self.teacher, teacher_ckpt_path)

    def load_model_weights(self, model, ckpt_path):
        config = self.config

        if 'fsdp' in config.train.strategy:
            if self.trainer.local_rank != 0:
                return

        print("Loading ", ckpt_path)       
        if 'deepspeed_stage_3' in config.train.strategy and deepspeed.comm.get_rank() != 0:
            load_dict = None
        else:
            if ckpt_path.lower().endswith('.safetensors'):
                load_dict = load_file(ckpt_path, device='cpu')
            else:
                load_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                
            # FIXME - this provides copies of tied weights, which isn't desirable for all models or when we want them to actually be tied
            if 'lm_head.weight' not in load_dict:
                load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
                
            # FIXME - this gives the inline teacher the copies it needs of the self_attn weights
            if config.train.attention_distillation_stage == 1:
                keys = list(load_dict.keys())
                for k in keys:
                    if '.self_attn.' in k:
                        load_dict[k.replace('self_attn', 'teacher_attn')] = load_dict[k]                            

        strict = not config.train.load_partial and config.train.attention_distillation_stage != 3

        if 'deepspeed_stage_3' not in config.train.strategy:
            model.load_state_dict(load_dict, strict=strict)
            print("Loaded ", ckpt_path)       
            return

        # # simple version that takes a lot more CPU RAM
        # self.trainer.strategy.model_to_device()
        # with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
        #     if deepspeed.comm.get_rank() == 0:
        #         model.load_state_dict(load_dict, strict=False)

        # print("Loaded ", ckpt_path)       
        # return

        # see https://github.com/microsoft/DeepSpeed/blob/8cded575a94e296fee751072e862304676c95316/deepspeed/runtime/zero/partition_parameters.py#L2172
        # see trainer.strategy.load_model_state_dict https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/strategies/deepspeed.html

        """Overrides the normal load_state_dict behaviour in PyTorch to ensure we gather parameters that may be sharded
        across processes before loading the state dictionary when using ZeRO stage 3. This is then automatically synced
        across processes.

        Args:
            ckpt: The ckpt file.

        """

        #assert self.lightning_module is not None

        def load(module: torch.nn.Module, prefix: str = "") -> None:
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    print("loading prefix", prefix)
                    missing_keys = []
                    unexpected_keys = []
                    error_msgs = []

                    # copy state_dict so _load_from_state_dict can modify it
                    metadata = getattr(load_dict, "_metadata", None)
                    state_dict = load_dict.copy()
                    if metadata is not None:
                        state_dict._metadata = metadata

                    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                
                    for n, p in module.named_parameters(recurse=False):
                        nn = prefix + n
                        print(nn, nn in state_dict)
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=strict,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )
                    if len(error_msgs) > 0:
                        print("ERROR", error_msgs)
                        exit(0)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")
        print("Loaded ", ckpt_path)       

    def forward(self, idx, last_model_state:ModelState|None = None):
        return self.model.forward(idx, last_model_state)
    
    def configure_optimizers(self):
        # what the heck, we had to do this before the optimizers are loaded or the loaded weights wouldn't 'take' into the optimizer!!!
        if 'deepspeed_stage_3' in self.config.train.strategy:
            self.load_weights()

        train_config = self.config.train

        optim_groups = self.model.get_optim_groups()

        print("Configuring optimizers!!!")

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

        B, T = x.shape
        causal_mask = torch.full((T, T), fill_value=-torch.inf, dtype=torch.bfloat16, device=x.device).triu(1)
        causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)

        if self.training and self.config.train.attention_distillation_stage in (1, 2):
            stage = self.config.train.attention_distillation_stage
            output_attentions = stage == 1
            output_post_attention_hidden_states = stage == 2
            # special code for attention output and/or attention matrix loss
            if self.config.model.classname != '':
                results = self.model.forward(x, output_hidden_states=False, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)
            else:
                results = self.model.forward(x, return_dict=False, attention_mask=causal_mask, output_hidden_states=False, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)

            if stage == 1:
                reported_loss = training_loss = torch.linalg.matrix_norm(torch.cat(results.attentions, dim=0) - torch.cat(results.student_attentions, dim=0)).mean() / results.attentions[0].size(-1)
            else: # stage == 2:
                reported_loss = training_loss = torch.linalg.vector_norm(torch.cat(results.post_attention_hidden_states, dim=0) - torch.cat(results.student_post_attention_hidden_states, dim=0), dim=-1).mean() * (results.post_attention_hidden_states[0].size(-1) ** -0.5)
            logits = torch.tensor([], device=x.device)
            preds = torch.zeros_like(y)
            return reported_loss, training_loss, logits, preds, last_model_state

        if self.config.model.classname != '':
            results = self.model(x, last_model_state, output_hidden_states=False)
        elif self.config.model.tmix.lower().startswith('qwen2'):
            results = self.model(x, attention_mask=causal_mask, output_hidden_states=False)
        else:
            results = self.model(x, last_model_state)
        if isinstance(results, tuple):
            logits = results[0]
            next_model_state = results[1]
        elif isinstance(results, torch.Tensor):
            logits = results
            next_model_state = last_model_state
        else:
            logits = results.logits
            next_model_state = last_model_state
    
        reported_loss = training_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.flatten())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)

        if self.training and self.teacher is not None:
            with torch.no_grad():
                teacher_results = self.teacher.forward(x)
                if isinstance(teacher_results, tuple):
                    teacher_logits = teacher_results[0]
                elif isinstance(results, torch.Tensor):
                    teacher_logits = teacher_results
                else:
                    teacher_logits = teacher_results.logits
            distillation_loss = F.kl_div(
                F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1),
                F.log_softmax(teacher_logits.view(-1, logits.size(-1)), dim=-1),
                log_target=True,
                reduction='batchmean'
            )
            training_loss = distillation_loss * self.config.train.teacher.kl_weight
            if self.config.train.teacher.ce_weight > 0:
                training_loss = training_loss + reported_loss * self.config.train.teacher.ce_weight

        if reported_loss.isinf().any():
            raise Exception("reported loss was infinite")

        if reported_loss.isnan().any():
            raise Exception("reported loss was NaN")

        if training_loss.isinf().any():
            raise Exception("loss was infinite")

        if training_loss.isnan().any():
            raise Exception("loss was NaN")

        return reported_loss, training_loss, logits, preds, next_model_state
    
    def get_real_global_step(self): return int(self.trainer.global_step + self.config.train.epoch_begin * self.config.runtime.epoch_global_steps)
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

        loss, training_loss, logits, preds, model_state = self._get_loss_logits_preds((inputs, labels), batch_idx, model_state)
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

        #if logits.size(0) > 0:
        #    return L2Wrap.apply(training_loss, logits)
        #else:
        return training_loss

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
        loss, training_loss, logits, preds, next_block_states = self._get_loss_logits_preds(batch, batch_idx, None)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for name, metric in self.metrics.items():
            metric.update(margs)
            # on_epoch causes this to be logged in aggregate rather than per batch
            #self.log('val/'+name, metric.compute(), on_epoch=True, rank_zero_only=True)
            #metric.clear()
        #self.log("tokens", float(self.all_nodes_tokens_processed), on_epoch=True, rank_zero_only=True)
        return loss
    
    def predict_dataloader(self):
        return DataLoader(mnist_predict, batch_size=self.batch_size)
    
    # def training_step_end(self, batch_parts):
    #     if pl.__version__[0]!='2':
    #         all = self.all_gather(batch_parts)
    #         if self.trainer.is_global_zero:
    #             self.trainer.my_loss_all = all
    