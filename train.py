from dataclasses import dataclass
from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config
import logging
from functools import partial
from typing import Callable

logging.basicConfig(level=logging.INFO)

from src.logger import print0 as print
from accelerate import init_empty_weights as init_on_meta_device

if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning import Trainer
    from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
    import lightning as pl
    from lightning.pytorch.strategies.fsdp import FSDPStrategy

    from transformers import AutoModelForCausalLM

    from contextlib import nullcontext, contextmanager

    rank_zero_info("########## work in progress ##########")

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    from pydoc import locate
  
    config, errors = parse_cmdline_configs(sys.argv[1:])
    if errors != '':
        print(errors)
        exit()
    

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    runtime_config = Runtime_Config()
    config.runtime = runtime_config
    runtime_config.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    runtime_config.global_step_bsz = int(config.train.num_nodes) * int(config.train.devices) * config.train.micro_bsz * config.train.accumulate_grad_batches
    os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
    os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
    os.environ["RWKV_ATTENTION_TYPE"] = config.model.attention_type

    model_name = f'{config.model.tmix}'
    if config.model.tmix2 != '':
        model_name += f'_{config.model.tmix2}'
    runtime_config.run_name = f"{model_name} L{config.model.n_layer} D{config.model.n_embd} ctx{config.model.ctx_len} "
    
    if config.train.proj_name == '':
        config.train.proj_name = f'L{config.model.n_layer}-D{config.model.n_embd}-{config.model.tmix}'
        if config.model.tmix2 != '':
            config.train.proj_name += f'_{config.model.tmix2}'
    config.runtime.proj_path = config.train.proj_dir + '/'
    config.runtime.proj_path += config.train.proj_name    
    if config.train.proj_suffix0 != '':
        config.runtime.proj_path += f'-{config.train.proj_suffix0}'
    if config.train.proj_suffix != '':
        config.runtime.proj_path += f'-{config.train.proj_suffix}'
    if not os.path.exists(config.runtime.proj_path):
        os.makedirs(config.runtime.proj_path)

    assert config.train.train_stage >= -1

    EPOCH_SAMPLE_SIZE = 40320 # 2*3*4*5*6*7*8
    runtime_config.epoch_count = max(1, config.train.magic_prime // EPOCH_SAMPLE_SIZE)

    runtime_config.epoch_global_steps = EPOCH_SAMPLE_SIZE // runtime_config.global_step_bsz
    assert runtime_config.epoch_global_steps * runtime_config.global_step_bsz == EPOCH_SAMPLE_SIZE
    if config.train.load_model == 'auto' and config.train.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(config.runtime.proj_path):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = (p.split("-")[-1].split("."))[0]
                if p != "final":
                    if p == "init":
                        list_p += [-1]
                    else:
                        try:
                            list_p += [int(p)]
                        except:
                            pass
        list_p.sort()
        if len(list_p) == 0:
            print(f"No correctly named rwkv-*.pth file found in {config.runtime.proj_path}")
            exit(-1)
        max_p = list_p[-1]
        if len(list_p) > 1:
            runtime_config.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
        if max_p == -1:
            config.train.load_model = f"{config.runtime.proj_path}/rwkv-init.pth"
        else:
            config.train.load_model = f"{config.runtime.proj_path}/rwkv-{max_p}.pth"
        if config.train.warmup_steps < 0:
            config.train.warmup_steps = 10
        config.train.epoch_begin = max_p + 1

    samples_per_epoch = runtime_config.epoch_global_steps * runtime_config.global_step_bsz
    tokens_per_epoch = samples_per_epoch * config.model.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# Model {model_name} {config.train.precision.upper()} on {config.train.num_nodes}x{config.train.devices} {config.train.accelerator.upper()}, bsz {config.train.num_nodes}x{config.train.devices}x{config.train.micro_bsz}={runtime_config.global_step_bsz}, {config.train.strategy} {'with grad_cp' if config.train.grad_cp > 0 else ''}
#
# Data = {config.train.data_file} ({config.train.data_type}), ProjDir = {config.runtime.proj_path}
#
# Epoch = {config.train.epoch_begin} to {config.runtime.epoch_count - 1} (will continue afterwards), save every {config.train.epoch_save} epoch
#
# Each "epoch" = {runtime_config.epoch_global_steps} global steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {config.model.n_layer} n_layer, {config.model.n_embd} n_embd, {config.model.ctx_len} ctx_len
#
# Adam = lr {config.train.lr_init} to {config.train.lr_final}, warmup {config.train.warmup_steps} steps, beta {(config.train.beta1, config.train.beta2)}, eps {config.train.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found lightning {pl.__version__}, requires 2+
#
############################################################################
"""
    )
    rank_zero_info(str(vars(config)) + "\n")

    assert config.train.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16", 'jsonl']

    assert config.train.precision in ["32", "tf32", "16", "16-true", "16-mixed", "bf16", "bf16-true", "bf16-mixed"]
    os.environ["RWKV_FLOAT_MODE"] = config.train.precision
    if str(config.train.precision) == "32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if str(config.train.precision).startswith("16"):
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    if "deepspeed_stage_3" in config.train.strategy or "fsdp" in config.train.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if str(config.train.precision) == "32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    pl.seed_everything(config.train.seed_everything)

    ########################################################################################################

    from src.trainer import train_callback
    from src.dataset import MyDataset, MMapDataset

    from src.lit import LightningModelWrapper
    from src.model import Transformer
    from qwen2.configuration_qwen2 import Qwen2Config

    from safetensors.torch import load_file

    # NOTE - this import MUST come AFTER the JIT gets disabled above or that disabling won't take effect correctly for the model
    #import models.qwen2

    strategy_obj = config.train.strategy
    if 'fsdp' in config.train.strategy:
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e6))
        #auto_wrap_policy = {models.qwen2.Qwen2DecoderLayer}
        activation_checkpointing_policy = None #{ models.qwen2.Qwen2DecoderLayer }

        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
        mixed_precision = None
        #mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        #mixed_precision = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        def init_fn(x: torch.nn.Module):
            return x.to_empty(device=torch.get_default_device(), recurse=False)

        strategy_obj = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing_policy=activation_checkpointing_policy, sync_module_states=True, mixed_precision=mixed_precision, limit_all_gathers=True, use_orig_params=True, param_init_fn=init_fn)


    # FIXME - why use_distributed_sampler=False? was this an oversight in the original repo? is this related to replace_sampler_ddp from Bo's code?
    trainer = Trainer(
                        use_distributed_sampler=config.train.data_type == 'jsonl',
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                        logger=False,
                        max_epochs=-1,

                        accelerator=config.train.accelerator, 
                        strategy=strategy_obj, 
                        devices=config.train.devices, 
                        num_nodes=config.train.num_nodes, 
                        precision=config.train.precision,
                        callbacks=[train_callback(config)], 
                        check_val_every_n_epoch=config.train.check_val_every_n_epoch, 
                        log_every_n_steps=config.train.log_every_n_steps, 
                        accumulate_grad_batches=config.train.accumulate_grad_batches, 
                        gradient_clip_val=config.train.gradient_clip_val, 
                        val_check_interval=config.train.val_check_interval)

    qwen_cfg = {
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151643,
        "hidden_act": "silu",
        "hidden_size": 896,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 131072,
        "max_window_layers": 24,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.1",
        "use_cache": False, #True,
        "use_sliding_window": False,
        "vocab_size": 151936,
    }

    @contextmanager
    def dualcontextmanager(a, b):
        with a(), b():
            yield

    @contextmanager
    def default_dtype_manager(dtype: torch.dtype):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        yield
        torch.set_default_dtype(old_dtype)

    ctx = nullcontext

    teacher = None
    teacher_factory = None
    teacher_config = None
    if config.train.train_stage > 1:
        teacher_config = config.train.teacher
        if teacher_config is not None and teacher_config.path != '':
            print("Instantiating teacher model")
            hf_path = teacher_config.model.hf_path
            classname = teacher_config.model.classname
            if hf_path != '':
                teacher = AutoModelForCausalLM.from_pretrained(hf_path)
            elif classname != '':
                teacher_classpath = f'models.{classname}.Model_{classname}'
                teacher_factory = locate(teacher_classpath)
                if teacher_factory is None:
                    print(f"Unsupported teacher model type: {teacher_classpath}")
                    exit(0)
                with ctx():
                    teacher = teacher_factory(teacher_config)
            elif teacher_config.model.tmix.startswith('qwen2'):
                from qwen2.modeling_qwen2 import Qwen2ForCausalLM
                teacher = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in teacher_config.model.tmix, **qwen_cfg), teacher_config)
            else:
                teacher = Transformer(teacher_config)

    if True:
        print("Instantiating student model")
        attention_distillation_stage = config.train.attention_distillation_stage
        hf_path = config.model.hf_path
        classname = config.model.classname
        if hf_path != '':
            ReplacementSelfAttnType = locate(config.model.attn_path)
            assert isinstance(ReplacementSelfAttnType, Callable)

            from rwkv6attn import load_and_patch_model_with_attention_replacement
            model = load_and_patch_model_with_attention_replacement(hf_path, config.model.attn_classes_path, ReplacementSelfAttnType, attention_distillation_stage)
        elif classname != '':
            model_classpath = f'models.{classname}.Model_{classname}'
            model_factory = locate(model_classpath)
            if model_factory is None:
                print(f"Unsupported model type: {model_classpath}")
                exit(0)
            with ctx():
                model = model_factory(config)
        elif config.model.tmix.startswith('qwen2'):
            from qwen2.modeling_qwen2 import Qwen2ForCausalLM
            model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **qwen_cfg), config)
        else:
            model = Transformer(config)

    if config.train.train_stage == 1:  # should we build the initial weights?
        if classname != '' or config.model.tmix.startswith("qwen2"):
            model.configure_model()
            mm = {k: v.cpu() for k, v in model.state_dict().items()} #model.state_dict()
        else:
            mm = model.generate_init_weight()
        init_weight_name = f"{config.runtime.proj_path}/rwkv-init.pth"
        print(f"Save to {init_weight_name}...")
        torch.save(mm, init_weight_name)
        print("Done. Now go for stage 2.")
        exit(0)

    # if trainer.global_rank == 0:
    #     for n in model.state_dict():
    #         shape = model.state_dict()[n].shape
    #         shape = [i for i in shape if i != 1]
    #         if len(shape) > 1:
    #             print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
    #         else:
    #             print(f"{str(shape[0]).ljust(5)}       {n}")

    if classname != '':
        pass
    elif config.model.tmix.startswith('qwen2'):
        if config.train.grad_cp:
            print("Setting grad cp")
            if "deepspeed" in config.train.strategy:
                model._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
                model.gradient_checkpointing = True
            else:
                model.gradient_checkpointing_enable()

        if os.getenv("RWKV_TORCH_COMPILE", '0').lower() in ['1', 'true']:
            print("Compiling model")
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)
        #elif os.getenv("RWKV_JIT_ON", '1').lower() in ['1', 'true']:
        #    model = torch.jit.script(model)
        #    if teacher is not None:
        #        teacher = torch.jit.script(teacher)

    print("Instantiating wrapper model")

    #with trainer.init_module(empty_init=not config.train.load_partial):
    wrapper = LightningModelWrapper(model, config, teacher, trainer) # delay setting the teacher until after init so deepspeed_stage_3 doesn't break it

    print("Creating DataLoader")

    if "deepspeed" in config.train.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000

    validation_data_loader = None


    if config.train.data_type == 'jsonl':
        import torch.utils.data
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        from raw_dataset import load_datasets_from_directories,TypedDataset,TypedStreamingCLMDataCollator
        all_ds,feature_types = load_datasets_from_directories(config.train.data_file.split(','),tokenizer)
        typed_dataset = TypedDataset(all_ds, feature_types)
        data_collator = TypedStreamingCLMDataCollator(tokenizer=tokenizer, 
                                                    max_length=config.model.ctx_len, 
                                                    min_length=config.model.ctx_len, 
                                                    typed_dataset=typed_dataset,
                                                    packing=True) # FIXME args.need_to_pad)
        # from torch.utils.data.distributed import DistributedSampler
        # train_sampler = DistributedSampler(
        #     typed_dataset,
        #     num_replicas=self.trainer.world_size,
        #     rank=self.trainer.local_rank,
        #     shuffle=True
        # )
        train_data_loader = torch.utils.data.DataLoader(
            typed_dataset, 
            batch_size=config.train.micro_bsz,
            #sampler=train_sampler,
            num_workers=1, 
            pin_memory=True, 
            persistent_workers=False, 
            drop_last=True, 
            collate_fn=data_collator,
            shuffle=True,
        ) 

    else:
        train_data = MyDataset(config, trainer)
        if config.train.validation_data_file != "":
            validation_data = MMapDataset(config.train.validation_data_file, config.model.ctx_len)
        config.model.vocab_size = train_data.vocab_size

        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        train_data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
        if config.train.validation_data_file != "":
            validation_data_loader = DataLoader(validation_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    ds_ckpt_path = None
    #if 'deepspeed_stage_3' in config.train.strategy and config.continued:
    #    ds_ckpt_path = config.train.load_model
    print("Trainer.fit")
    trainer.fit(wrapper, train_dataloaders=train_data_loader, val_dataloaders=validation_data_loader, ckpt_path=ds_ckpt_path)
    #trainer.fit(wrapper, ckpt_path=ds_ckpt_path)

