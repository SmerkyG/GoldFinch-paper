########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import transformers # just for a bugfix for 0.4.2 of lm_eval

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

from pydoc import locate

from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from src.pipeline import PIPELINE, PIPELINE_ARGS

from lm_eval import tasks, evaluator, utils
from lm_eval.models.huggingface import HFLM

from qwen2.modeling_qwen2 import Qwen2ForCausalLM
from qwen2.configuration_qwen2 import Qwen2Config

from tqdm import tqdm

########################################################################################################

from dataclasses import dataclass
import typing

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    tasks: str = 'lambada_openai' # arc_challenge, arc_easy, headqa, openbookqa, hellaswag, winogrande, piqa, record, copa, storycloze_2016
    bsz: int = 48
    precision: int | str = 'bf16'
    seed: int | None = None
    recurrent: int = 1
    train:typing.Any = None
    model: Model_Config

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()

os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)

model_path = config.path

# Setup the model
from src.model import Transformer
from safetensors.torch import load_file

print(f'Loading model - {model_path}')
classname = config.model.classname
if config.path.lower().endswith('.safetensors'):
    load_dict = load_file(config.path)
else:
    load_dict = torch.load(model_path, mmap=True)
if classname.startswith('qwen2') or config.model.tmix.startswith('qwen2'):
    load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
    
#with torch.device('meta'):
if True:
    if config.model.hf_cfg is not None: #tmix.startswith('qwen2'):
        model_classpath = classname
        model_factory = locate(model_classpath)
        if model_factory is None:
            print(f"Unsupported model type: {model_classpath}")
            exit(0)
        #config_class = model_factory.config_class
        print(config.model.hf_cfg)
        model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **config.model.hf_cfg), config)
        #model = model_factory(config_class(**hf_config))
    elif classname != '':
        if '.' in classname:
            model_classpath = classname
        else:
            model_classpath = f'models.{classname}.Model_{classname}'
        model_factory = locate(model_classpath)
        if model_factory is None:
            print(f"Unsupported model type: {model_classpath}")
            exit(0)
        model = model_factory(config)
    else:
        model = Transformer(config)
model.load_state_dict(load_dict, assign=True)

match config.precision:
    case 32:
        dtype = torch.float32
    case '32':
        dtype = torch.float32
    case 16:
        dtype = torch.float16
    case '16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16
    case _:
        print("Bad precision type specified")
        exit()

device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()

eval_tasks = config.tasks.split(',')

########################################################################################################

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        #self.eos_token_id = 0
        self.eos_token_id = tokenizer.eos_token_id

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

if config.seed is None:
    config.seed = 1234 

# tokenizer = TokenizerWrapper(pipeline.tokenizer) # RWKV tokenizer
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
tokenizer:transformers.PreTrainedTokenizer = Qwen2Tokenizer.from_pretrained('Qwen/Qwen-tokenizer')

adapter = HFLM(pretrained=model, tokenizer=tokenizer)
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
	    results = evaluator.simple_evaluate(
	        model=adapter,
	        tasks=eval_tasks,
	        #provide_description=False,
	        num_fewshot=0,
	        limit=None,
	        bootstrap_iters=10000,
	        numpy_random_seed = config.seed,
	        torch_random_seed = config.seed,
	        # fewshot_random_seed = config.seed, # FIXME - needed in next version of lm_eval
	    )

print(results['results'])
