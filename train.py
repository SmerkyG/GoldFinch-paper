from dataclasses import dataclass
from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from lightning import Trainer
    from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
    import lightning as pl

    rank_zero_info("########## work in progress ##########")

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from pydoc import locate
  
    config, errors = parse_cmdline_configs(sys.argv[1:])
    if errors != '':
        print(errors)
        exit()
    

    if "deepspeed" in config.train.strategy:
        import deepspeed

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
    if config.train.proj_suffix != '':
        config.runtime.proj_path += f'-{config.train.proj_suffix}'
    if not os.path.exists(config.runtime.proj_path):
        os.makedirs(config.runtime.proj_path)

    assert config.train.train_stage >= -1

    EPOCH_SAMPLE_SIZE = 40320
    runtime_config.epoch_count = config.train.magic_prime // EPOCH_SAMPLE_SIZE

    runtime_config.epoch_global_steps = EPOCH_SAMPLE_SIZE // runtime_config.global_step_bsz
    assert runtime_config.epoch_global_steps * runtime_config.global_step_bsz == EPOCH_SAMPLE_SIZE
    if config.train.load_model == '' and config.train.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(config.runtime.proj_path):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
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

    assert config.train.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    assert config.train.precision in ["32", "tf32", "16", "16-true", "16-mixed", "bf16", "bf16-true", "bf16-mixed"]
    os.environ["RWKV_FLOAT_MODE"] = config.train.precision
    if str(config.train.precision) == "32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if str(config.train.precision).startswith("16"):
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    if "deepspeed_stage_3" in config.train.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if str(config.train.precision) == "32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    pl.seed_everything(config.train.seed_everything)

    ########################################################################################################

    from src.trainer import train_callback
    from src.dataset import MyDataset, MMapDataset

    from src.lit import LightningModelWrapper
    from src.model import Transformer
    from qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from qwen2.configuration_qwen2 import Qwen2Config

    from safetensors.torch import load_file

    # FIXME - why use_distributed_sampler=False? was this an oversight in the original repo? is this related to replace_sampler_ddp from Bo's code?
    trainer = Trainer(
                        use_distributed_sampler=False, 
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                        logger=False,
                        max_epochs=-1,

                        accelerator=config.train.accelerator, 
                        strategy=config.train.strategy, 
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

    teacher = None
    if config.train.train_stage > 1:
        teacher_config = config.train.teacher
        if teacher_config is not None and teacher_config.path != '':
            classname = teacher_config.model.classname
            with trainer.init_module(empty_init=True):
                if classname != '':
                    teacher_classpath = f'models.{classname}.Model_{classname}'
                    teacher_factory = locate(teacher_classpath)
                    if teacher_factory is None:
                        print(f"Unsupported teacher model type: {teacher_classpath}")
                        exit(0)
                    teacher = teacher_factory(teacher_config)
                elif teacher_config.model.tmix.startswith('qwen2'):
                    teacher = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in teacher_config.model.tmix, **qwen_cfg), teacher_config)
                else:
                    teacher = Transformer(teacher_config)
            # FIXME - hacked in weight tying
            if classname.startswith('qwen2') or config.model.tmix.startswith('qwen2'):
                teacher.lm_head.weight = teacher.model.embed_tokens.weight
            teacher = teacher.to(torch.bfloat16) # NOTE - doing this here because otherwise it doesn't get done when not using deepspeed

            if teacher_config.path.lower().endswith('.safetensors'):
                load_dict = load_file(teacher_config.path)
            else:
                load_dict = torch.load(teacher_config.path, map_location="cpu")
            # FIXME - hacked in weight tying
            if classname.startswith('qwen2') or config.model.tmix.startswith('qwen2'):
                load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
            teacher.load_state_dict(load_dict)
            teacher.eval()
            teacher.requires_grad_(False)

    with trainer.init_module(empty_init=not config.train.load_partial):
        classname = config.model.classname
        if classname != '':
            model_classpath = f'models.{classname}.Model_{classname}'
            model_factory = locate(model_classpath)
            if model_factory is None:
                print(f"Unsupported model type: {model_classpath}")
                exit(0)
            model = model_factory(config)
        elif config.model.tmix.startswith('qwen2'):
            model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **qwen_cfg), config)
        else:
            model = Transformer(config)
        # FIXME - hacked in weight tying
        if classname.startswith('qwen2') or config.model.tmix.startswith('qwen2'):
            model.lm_head.weight = model.model.embed_tokens.weight
                
    if config.train.train_stage == 1:  # should we build the initial weights?
        init_weight_name = f"{config.runtime.proj_path}/rwkv-init.pth"
        if classname != '':
            model.apply(model._init_weights)
            model.init_all_weights()
            mm = {k: v.cpu() for k, v in model.state_dict().items()} #model.state_dict()
        elif config.model.tmix.startswith("qwen2"):
            model.apply(model._init_weights)
            model.init_all_weights()
            mm = {k: v.cpu() for k, v in model.state_dict().items()} #model.state_dict()
        else:
            mm = model.generate_init_weight()
        print(f"Save to {init_weight_name}...")
        torch.save(mm, init_weight_name)
        print("Done. Now go for stage 2.")
        exit(0)

    if config.train.train_stage == 0 or config.train.load_partial == 1:
        if classname != '':
            model.apply(model._init_weights)
            model.init_all_weights()
        elif config.model.tmix.startswith("qwen2"):
            model.apply(model._init_weights)
            model.init_all_weights()
        #else:
            #mm = model.init_weights() # already done in the constructor

    if config.train.load_model != '':#config.train.train_stage >= 2:
        rank_zero_info(f"########## Loading {config.train.load_model}... ##########")
        if config.train.load_model.lower().endswith('.safetensors'):
            load_dict = load_file(config.train.load_model)
        else:
            load_dict = torch.load(config.train.load_model, map_location="cpu")
        # FIXME - hacked in weight tying
        if classname.startswith('qwen2') or config.model.tmix.startswith('qwen2'):
            load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']

        if config.train.load_partial == 1:
            load_keys = load_dict.keys()
            for k in model.state_dict():
                if k not in load_keys:
                    load_dict[k] = model.state_dict()[k]
        model.load_state_dict(load_dict, strict = not config.train.load_partial)

    if config.train.train_stage == -1:
        init_weight_name = f"{config.runtime.proj_path}/rwkv-init.pth"
        mm = {k: v.cpu() for k, v in model.state_dict().items()} #model.state_dict()
        print(f"Save to {init_weight_name}...")
        torch.save(mm, init_weight_name)
        print("Done. Now go for stage 2.")
        exit(0)

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            shape = [i for i in shape if i != 1]
            if len(shape) > 1:
                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
            else:
                print(f"{str(shape[0]).ljust(5)}       {n}")

    if "deepspeed" in config.train.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000

    if classname != '':
        pass
    elif config.model.tmix.startswith('qwen2'):
        if config.train.grad_cp:
            if "deepspeed" in config.train.strategy:
                model._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
                model.gradient_checkpointing = True
            else:
                model.gradient_checkpointing_enable()

        if os.getenv("RWKV_TORCH_COMPILE", '0').lower() in ['1', 'true']:
            model = torch.compile(model)
            if teacher is not None:
                teacher = torch.compile(teacher)
        #elif os.getenv("RWKV_JIT_ON", '1').lower() in ['1', 'true']:
        #    model = torch.jit.script(model)
        #    if teacher is not None:
        #        teacher = torch.jit.script(teacher)

    with trainer.init_module(empty_init=not config.train.load_partial):
        wrapper = LightningModelWrapper(model, config, teacher)
       
    train_data = MyDataset(config, trainer)
    if config.train.validation_data_file != "":
        validation_data = MMapDataset(config.train.validation_data_file, config.model.ctx_len)
    config.model.vocab_size = train_data.vocab_size

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    train_data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    validation_data_loader = None
    if config.train.validation_data_file != "":
        validation_data_loader = DataLoader(validation_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    trainer.fit(wrapper, train_dataloaders=train_data_loader, val_dataloaders=validation_data_loader)
