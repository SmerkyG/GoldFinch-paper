from dataclasses import dataclass
from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config
import lightning.pytorch.callbacks
import logging

logging.basicConfig(level=logging.INFO)

from lightning.fabric.utilities.types import _PATH
from typing import Any, Dict, Optional, Set
from datetime import timedelta
class ModelCheckpoint(lightning.pytorch.callbacks.ModelCheckpoint):

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
    ):
        super().__init__(
            dirpath,
            filename,
            monitor,
            verbose,
            save_last,
            save_top_k,
            save_weights_only,
            mode,
            auto_insert_metric_name,
            every_n_train_steps,
            train_time_interval,
            every_n_epochs,
            save_on_train_epoch_end,
            enable_version_counter,
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                # NOTE - moved this over so we don't save when not the correct epoch
                self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._should_skip_saving_checkpoint(trainer) and not self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
                # NOTE - moved this over so we don't save when not the correct epoch
                self._save_last_checkpoint(trainer, monitor_candidates)
    

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
    from torchdata.stateful_dataloader import StatefulDataLoader
  
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

    assert config.train.train_stage > 0

    EPOCH_SAMPLE_SIZE = 40320
    runtime_config.epoch_count = config.train.magic_prime // EPOCH_SAMPLE_SIZE

    runtime_config.epoch_global_steps = EPOCH_SAMPLE_SIZE // runtime_config.global_step_bsz
    assert runtime_config.epoch_global_steps * runtime_config.global_step_bsz == EPOCH_SAMPLE_SIZE
    # if config.train.train_stage >= 2:  # find latest saved model
    #     list_p = []
    #     for p in os.listdir(config.runtime.proj_path):
    #         if p.startswith("rwkv") and p.endswith(".pth"):
    #             p = ((p.split("-"))[1].split("."))[0]
    #             if p != "final":
    #                 if p == "init":
    #                     p = -1
    #                 else:
    #                     p = int(p)
    #                 list_p += [p]
    #     list_p.sort()
    #     if len(list_p) == 0:
    #         print(f"No correctly named rwkv-*.pth file found in {config.runtime.proj_path}")
    #         exit(-1)
    #     max_p = list_p[-1]
    #     if len(list_p) > 1:
    #         runtime_config.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
    #     if max_p == -1:
    #         config.train.load_model = f"{config.runtime.proj_path}/rwkv-init.pth"
    #     else:
    #         config.train.load_model = f"{config.runtime.proj_path}/rwkv-{max_p}.pth"
    if config.train.warmup_steps < 0:
        config.train.warmup_steps = 10
    #     config.train.epoch_begin = max_p + 1

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
# Epoch =  to {config.runtime.epoch_count - 1} (will continue afterwards), save every {config.train.save_every_n_epochs} epoch, {config.train.save_every_n_steps} steps
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

    # FIXME - why use_distributed_sampler=False? was this an oversight in the original repo? is this related to replace_sampler_ddp from Bo's code?
    trainer = Trainer(
                        use_distributed_sampler=False, 
                        enable_checkpointing=True,
                        num_sanity_val_steps=0,
                        logger=False,
                        max_epochs=config.runtime.epoch_count,

                        accelerator=config.train.accelerator, 
                        strategy=config.train.strategy, 
                        devices=config.train.devices, 
                        num_nodes=config.train.num_nodes, 
                        precision=config.train.precision,
                        callbacks=[
                            train_callback(config),
                            ModelCheckpoint(
                                every_n_epochs=config.train.save_every_n_epochs,
                                every_n_train_steps=config.train.save_every_n_steps,
                                save_on_train_epoch_end=True,
                                #monitor='loss',
                                save_last=True,
                                monitor='step',
                                mode='max',
                                save_top_k=config.train.save_top_k,
                                dirpath=f'{config.runtime.proj_path}/',
                                #filename='rwkv-{epoch:02d}-{loss:.2f}'
                                filename='{epoch:03d}-{step}',
                            ),
                        ], 
                        check_val_every_n_epoch=config.train.check_val_every_n_epoch, 
                        log_every_n_steps=config.train.log_every_n_steps, 
                        accumulate_grad_batches=config.train.accumulate_grad_batches, 
                        gradient_clip_val=config.train.gradient_clip_val, 
                        val_check_interval=config.train.val_check_interval)

    with trainer.init_module(empty_init=not config.train.load_partial):
        model = Transformer(config)
        wrapper = LightningModelWrapper(model, config)

    if config.train.train_stage == 1:  # should we build the initial weights?
        init_weight_name = f"{config.runtime.proj_path}/rwkv-init.pth"
        mm = model.generate_init_weight()
        print(f"Save to {init_weight_name}...")
        torch.save(mm, init_weight_name)
        print("Done. Now go for stage 2.")
        exit(0)

    if config.train.ckpt_path is None:
        if os.path.isdir(f'{config.runtime.proj_path}/last.ckpt'):
            config.train.ckpt_path = f'{config.runtime.proj_path}/last.ckpt'
        elif os.path.exists(f"{config.runtime.proj_path}/rwkv-init.pth"):
            config.train.ckpt_path = f"{config.runtime.proj_path}/rwkv-init.pth"
        else:
            print(f"Please run with train_stage 1 to generate rwkv-init.pth - no last.ckpt or rwkv-init.pth exists at path {config.runtime.proj_path}")
            exit(0)

    if config.train.ckpt_path.lower().endswith('.pth'):
        rank_zero_info(f"########## Loading {config.train.ckpt_path}... ##########")
        load_dict = torch.load(config.train.ckpt_path, map_location="cpu")

        # if config.train.load_partial == 1:
        #     load_keys = load_dict.keys()
        #     for k in model.state_dict():
        #         if k not in load_keys:
        #             load_dict[k] = model.state_dict()[k]

        model.load_state_dict(load_dict, strict = not config.train.load_partial)

        config.train.ckpt_path = None

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

    train_data = MyDataset(config, trainer)
    if config.train.validation_data_file != "":
        validation_data = MMapDataset(config.train.validation_data_file, config.model.ctx_len)
        if config.train.validation_tokens > 0 and config.train.validation_tokens // (config.model.ctx_len + 1) < len(validation_data):
            from torch.utils.data import Subset
            validation_data = Subset(validation_data, range(config.train.validation_tokens // config.model.ctx_len))
    config.model.vocab_size = train_data.vocab_size

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    train_data_loader = StatefulDataLoader(train_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)
    validation_data_loader = None
    if config.train.validation_data_file != "":
        validation_data_loader = DataLoader(validation_data, shuffle=False, pin_memory=True, batch_size=config.train.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    trainer.fit(wrapper, train_dataloaders=train_data_loader, val_dataloaders=validation_data_loader, ckpt_path=config.train.ckpt_path)
