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
    def __init__(self, model:nn.Module, config:TrainerCLI_Config, teacher:nn.Module|None):
        super().__init__()
        self.model = model
        self.config = config
        self.teacher = teacher
        self.metrics = dict(loss=metrics.Loss(), acc=metrics.Accuracy())

    def configure_model(self):
        if hasattr(self.model, 'configure_model'):
            self.model.configure_model()
        self.init_all_weights()

    def init_all_weights(self):
        if hasattr(self.model, 'init_all_weights'):
            self.model.init_all_weights()

    def load_weights(self):
        ckpt_path = self.config.train.load_model
        if ckpt_path.endswith('/latest'):
            # model already loaded as part of lightning
            return

        print("Loading ", ckpt_path)
        if ckpt_path.lower().endswith('.safetensors'):
            load_dict = load_file(ckpt_path, device='cpu')
        else:
            load_dict = torch.load(ckpt_path, map_location='cpu')
            
        # FIXME - this provides copies of tied weights, which isn't desirable for all models or when we want them to actually be tied
        if 'lm_head.weight' not in load_dict:
            load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
            
        # FIXME - this gives the inline teacher the copies it needs of the self_attn weights
        if self.config.train.attention_distillation_stage == 1:
            keys = list(load_dict.keys())
            for k in keys:
                if '.self_attn.' in k:
                    load_dict[k.replace('self_attn', 'teacher_attn')] = load_dict[k]                            

        strict = False

        model = self.model
        #with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
        #    if deepspeed.comm.get_rank() == 0:
        #        model.load_state_dict(load_dict, strict=False)

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
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
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

    def load_weights_old(self):
        # FIXME - allow loading from sharded model so we use less CPU RAM and don't require conversion from safetensors

        ds3 = False
        if self.config.train is not None:
            if self.config.train.strategy == 'deepspeed_stage_3':
                ds3 = True

        # Load the model only on the master process (rank 0)
        if self.global_rank == 0 or not ds3:
            print("LIGHTNING RANK 0 = PYTORCH RANK", dist.get_rank())
            ckpt_path = self.config.train.load_model
            print("Loading ", ckpt_path, "on rank", self.global_rank)
            if ckpt_path.lower().endswith('.safetensors'):
                load_dict = load_file(ckpt_path, device='cpu')
            else:
                load_dict = torch.load(ckpt_path, map_location='cpu')
                
            # FIXME - this provides copies of tied weights, which isn't desirable for all models or when we want them to actually be tied
            if 'lm_head.weight' not in load_dict:
                load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
                
            # FIXME - this gives the inline teacher the copies it needs of the self_attn weights
            if self.config.train.attention_distillation_stage == 1:
                keys = list(load_dict.keys())
                for k in keys:
                    if '.self_attn.' in k:
                        load_dict[k.replace('self_attn', 'teacher_attn')] = load_dict[k]                            
                        
            for k in load_dict.keys():
                if k.startswith('_forward_module.'):
                    load_dict[k.replace('_forward_module.','')] = load_dict[k]
                    del load_dict[k]
    
            print("Loaded model on rank", self.local_rank)
            for n in load_dict:
                shape = load_dict[n].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 2:
                    print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(shape[2]).ljust(5)} {n}")
                elif len(shape) > 1:
                    print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)}       {n}")
                else:
                    print(f"{str(shape[0]).ljust(5)}             {n}")

            if not ds3:
                self.model.load_state_dict(load_dict, strict = not self.config.train.load_partial)
                del load_dict
                return
        else:
            load_dict = {}

        # Create a list to hold sizes for broadcast
        size_list = []
        
        # Prepare sizes list on rank 0
        if self.global_rank == 0:
            for k, v in load_dict.items():
                size_list.append((k, v.shape))
            
            # Pickle the size_list
            pickled_size_list = pickle.dumps(size_list)
        else:
            pickled_size_list = None
        
        dist.barrier()
        
        # Broadcast the pickled size_list from rank 0 to all other ranks
        pickled_size_list = self.trainer.strategy.broadcast(pickled_size_list, src=0)
        
        # Unpickle the size_list on all ranks
        size_list = pickle.loads(pickled_size_list)
        
        #print("Broadcasted tensor sizes list length:", len(size_list), "rank:", self.trainer.global_rank)
        

        size_dict = {}
        # Update load_dict on non-master ranks using the received sizes
        for item in size_list:
            size_dict[item[0]] = item[1]

        #print("Broadcasted tensor sizes length:", len(size_dict), "rank:", self.global_rank)
        sorted_keys = sorted(size_dict.keys())

        updated = 0
        count = 0
        print("\n" * 10)
        # Synchronize and broadcast the load_dict to all processes
        if self.global_rank == 0:
            print(f"Progress: - | -                              ", end="\r", flush=True)
        for name in sorted_keys:        
            if self.global_rank == 0:
                count += 1
                progress = f"{count}/{len(size_dict)}"
                print(f"Progress: {progress} | Sending name: {name}                              ", end="\r", flush=True)
                # Ensure all ranks have the tensor to be broadcasted
                tensor = load_dict[name].bfloat16().to(self.device)
            else:
                tensor = torch.zeros(size_dict[name], dtype=torch.bfloat16).to(self.device)
                tensor.view(-1)[0] = -torch.inf # have to set this for some really strange reason but pytorch broadcast is so much faster

                
            #tensor = self.trainer.strategy.broadcast(tensor, src=0)
            dist.broadcast(tensor, src=0)
            

            if tensor.shape != size_dict[name] or torch.sum(tensor) == -torch.inf:
                print(tensor)
                print("The broadcast failed for name: ", name)
                
            tensor = tensor.to(torch.bfloat16)
            tensor = tensor.to(self.device)

            #tensor = pickle.loads(tensor_pickle)
                
            
            #print(tensor.shape, name, self.global_rank)
            #self.configure_model()

            # idk why it needs to be loaded this way, but the other way below doesn't work so idk
            for n, p in self.named_parameters():
                if n == name:
                    #if p.shape == tensor.shape:
                    
                    if self.global_rank == 0:
                        print("Setting param")
                        
                    safe_set_full_fp32_param(p, tensor)
                    
                    if self.global_rank == 0:
                        print("Set param")
                    #safe_set_local_fp32_param(p, tensor.to(self.device))
                    updated += 1
                    break

            #if name in state_dict.keys():# and state_dict[name].shape == tensor.shape:
            #    safe_set_full_fp32_param(state_dict[name], tensor.to(self.device))
            #    updated += 1

            #print(self.global_rank, name, tensor.shape)
                    
            #p = self.state_dict()[name]
            #safe_set_full_fp32_param(p, tensor.cuda(self.local_rank))
            
            tensor = tensor.detach()
            del tensor
            torch.cuda.empty_cache()
        dist.barrier()
        
        print("\nUpdated", updated, "keys on", self.global_rank)

        tensor_diff_count = 0

        for n, p in self.named_parameters():
            if not torch.all(p == 0):
                tensor_diff_count += 1

        print("Non zero tensors rank", self.global_rank, "-", tensor_diff_count) 

        del load_dict
        torch.cuda.empty_cache()
        # Optionally handle partial loading of model parameters
        # Need to sort this out as it no longer works with the new distributed loading process.
        #if self.args.load_partial == 1:
        #    model_state_dict = self.state_dict()
        #    for k in model_state_dict:
        #        if k not in new_load_dict:
        #            new_load_dict[k] = model_state_dict[k].cuda(self.local_rank)

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
    