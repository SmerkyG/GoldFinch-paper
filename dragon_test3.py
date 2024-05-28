import os
import sys
import torch


# set these before import RWKV
#os.environ['RWKV_JIT_ON'] = '1'
#os.environ["RWKV_CUDA_ON"] = '0' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

class Args(dict):
    def __init__(self, **kwargs):
        for name, value in kwargs:
            self[name] = value
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e)
    def __setattr__(self, name, value):
         self[name] = value

args = Args()
args.head_size_a=64
args.head_size_divisor=8
args.ctx_len=2048
args.n_layer=24
args.n_embd=2048
args.dim_att=0
args.dim_ffn=0
args.vocab_size=65536
args.dropout=0.0
args.layerwise_lr=True
args.train_stage=2
args.grad_cp=True
args.betas=[0.9,0.99]
args.adam_eps=1e-8
#args..epoch_begin
#args.epoch_steps
#args.real_bsz
args.wandb=''
args.model_type = sys.argv[2] #'llama3'#'x060b2_poco'

args.load_partial=0
args.proj_dir=''


os.environ["RWKV_MODEL_TYPE"] = args.model_type
os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)


# Setup the model
import lightning as pl
#trainer = pl.Trainer(precision=32)
from src.model import SimpleRWKV, RWKV
#with trainer.init_module(empty_init=True):
#    model = RWKV(args)


MODEL_PATH=sys.argv[1]
model_path = MODEL_PATH

recurrent = sys.argv[3] == 'recurrent'

print(f"########## Loading {model_path}... ##########")
# try:
#     load_dict = torch.load(model_path, map_location="cpu")
#     load_keys = list(load_dict.keys())
#     for k in load_keys:
#         if k.startswith("_forward_module."):
#             load_dict[k.replace("_forward_module.", "")] = load_dict[k]
#             del load_dict[k]
# except:
#     print(f"Bad checkpoint {model_path}")
#     if args.train_stage >= 2:  # try again using another checkpoint
#         max_p = args.my_pile_prev_p
#         if max_p == -1:
#             model_path = f"{args.proj_dir}/rwkv-init.pth"
#         else:
#             model_path = f"{args.proj_dir}/rwkv-{max_p}.pth"
#         args.epoch_begin = max_p + 1
#         print(f"Trying {model_path}")
#         load_dict = torch.load(model_path, map_location="cpu")

# if args.load_partial == 1:
#     load_keys = load_dict.keys()
#     for k in model.state_dict():
#         if k not in load_keys:
#             load_dict[k] = model.state_dict()[k]

state_dict = torch.load(model_path, mmap=True)
with torch.device('meta'):
    model = RWKV(args)
model.load_state_dict(state_dict, assign=True)

#model.load_state_dict(load_dict)

dtype = torch.float32
device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()
if dtype != torch.float:
    torch.set_autocast_gpu_dtype(dtype)


from utils import PIPELINE, PIPELINE_ARGS

# download models: https://huggingface.co/BlinkDL
#model = RWKV(model='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023', strategy='cpu fp32')
#model = RWKV(model='out/L24-D2048-x060-0/rwkv-final.pth', strategy='cuda:4 fp16')
#pipeline = PIPELINE(model, "src/dataflow/20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
pipeline = PIPELINE(model, "rwkv_vocab_v20230424") # for rwkv "world" models

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
print(ctx, end='')

def my_print(s):
    print(s, end='', flush=True)

# For alpha_frequency and alpha_presence, see "Frequency and presence penalties":
# https://platform.openai.com/docs/api-reference/parameter-details

args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = [], # stop generation whenever you see any token here
                     chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

pl.seed_everything(1337)

pipeline.generate(ctx, token_count=200, args=args, callback=my_print, recurrent=recurrent)
print('\n')

# out, state = model.forward([187, 510, 1563, 310, 247], None)
# print(out.detach().cpu().numpy())                   # get logits
# out, state = model.forward([187, 510], None)
# out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
# out, state = model.forward([310, 247], state)
# print(out.detach().cpu().numpy())                   # same result as above
# print('\n')
