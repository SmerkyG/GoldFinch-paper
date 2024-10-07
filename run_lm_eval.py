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

from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from src.pipeline import PIPELINE, PIPELINE_ARGS

from src.logger import print0 as print

from lm_eval import tasks, evaluator
from lm_eval.api.model import TemplateLM

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

print(f'Loading model - {model_path}')
state_dict = torch.load(model_path, mmap=True)
with torch.device('meta'):
    model = Transformer(config)
model.load_state_dict(state_dict, assign=True)

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

pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

eval_tasks = config.tasks.split(',')

RWKV_PAD = pipeline.tokenizer.encode('\n') # we will use '\n' as PAD
# RWKV_PAD = [0] # you can try using [0] as pad
print('RWKV_PAD', RWKV_PAD)

########################################################################################################

logitBuf = {}
correctBuf = {}

class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = 0

    def encode(self, string: str, add_special_tokens=False):
        return self.tokenizer.encode(string)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

class EvalHarnessAdapter(TemplateLM):
    # bugfix for lm_eval 0.4.2
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, batch_size_per_gpu, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size_per_gpu

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported"
        )
    
    @torch.no_grad()
    def greedy_generate(self, ctx, state=None):
        STOP_TOKEN = [self.tokenizer.eos_token_id]

        all_tokens = []
        out_last = 0
        out_str = ''
        for i in range(self.max_gen_toks):
            tokens = self.tokenizer.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                results = model.forward(tokens[:self.max_length], state)
                if isinstance(results, tuple):
                    logits = results[0]
                    #next_model_state = results[1]
                elif isinstance(results, torch.Tensor):
                    logits = results
                    #next_model_state = last_model_state
                else:
                    logits = results.logits
                    #next_model_state = last_model_state
                tokens = tokens[self.max_length:]
            token = logits.argmax().item()
            if token in STOP_TOKEN:
                break
            all_tokens += [token]
            tmp = self.tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # is valid utf-8 string?
                out_str += tmp
                out_last = i + 1
        return out_str
    
    @torch.no_grad()
    def generate_until(self, requests):
        """
        Generate until is lm_eval harness' way to say "do greedy generation" - necessary for some tasks.
        the eval harness dispatches requests to the model, and the model does argmax generation, the results of which
        are returned to the eval harness to evaluate.

        TODO: batched / data parallel generation

        :param requests: Dictionary of requests containing the context (prompt) and 'until' - a token or
                         list of stop tokens.
        """
        res = []
        # get only the args from each Instance object
        reqs = [req.args for req in requests]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(reqs, _collate)
        for context, gen_kwargs in tqdm(reord.get_reordered(), "Running greedy generation"):
            out_str = self.greedy_generate(context)
            for term in gen_kwargs['until']:
                out_str = out_str.split(term)[0]
            res.append(out_str)
        return reord.get_original(res)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # FIXME - is this correct? is it even used? didn't seem to be
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len
        # try:
        #     return self.gpt2.config.n_ctx
        # except AttributeError:
        #     # gptneoconfig doesn't have n_ctx apparently
        #     return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        # FIXME - is this correct? is it even used? didn't seem to be, since the model itself complained at 512 when returning 256 here
        # FIXME - we really should support recurrent inference
        return config.model.ctx_len

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        global logitBuf, correctBuf

        # sort requests by descending total length, so we batch together groups that have similar padded sizes, descending so we OOM early if at all
        rq_indices = sorted(range(len(requests)),key=lambda i: len(requests[i][1])+len(requests[i][2]), reverse=True)

        res = [None for _ in range(len(requests))]

        B = self.batch_size_per_gpu
        for nb in range(0, len(requests), B):
            ne = min(nb+B, len(requests))

            # stack and pad to longest
            batched_inputs = []
            batch_info = []
            maxlen = 0
            for i in range(nb, ne):
                rq_index = rq_indices[i]
                request = requests[rq_index]
                q = RWKV_PAD + request[1]
                src = q + request[2]
                input = torch.tensor(src, dtype=torch.long, device=device, requires_grad=False)
                batched_inputs.append( input )
                batch_info.append((len(q), len(src), rq_index))
                maxlen = max(maxlen, len(src))

            maxlen = (maxlen + 7) // 8 * 8 # round pad size up to nearest 8 for better GPU usage
            for i in range(len(batched_inputs)):
                batched_inputs[i] = F.pad(batched_inputs[i], (0, maxlen - batched_inputs[i].size(0)))
            batched_inputs = torch.stack(batched_inputs, dim=0)

            logits, _ = model.forward(batch, None)


            batched_logprobs = F.log_softmax(logits, dim=-1)
            # Check if per-token argmax is exactly equal to continuation
            batched_greedy_toks = batched_logprobs.argmax(dim=-1)
            
            for i, info in enumerate(batch_info):
                q_len, src_len, rq_index = info
                a_len = src_len - q_len
                logprobs, a_toks, greedy_toks = batched_logprobs[i, q_len-1 : src_len-1], batched_inputs[i, q_len : src_len], batched_greedy_toks[i, q_len-1 : src_len-1]
                assert logprobs.size(0) == a_len
                assert a_toks.size(0) == a_len
                assert greedy_toks.size(0) == a_len
                max_equal = (greedy_toks == a_toks).all()
        
                # Obtain log-probs at the corresponding continuation ('answer') token indices
                logprobs = torch.gather(logprobs, 1, a_toks.unsqueeze(-1)).squeeze(-1)
                assert logprobs.size(0) == a_len
            
                # Answer: (log prob, is-exact-match)
                answer = (float(logprobs.sum()), bool(max_equal))

                # place the answer into the slot that matches the original request (this is important or lm_eval_harness will return bad results!!!)
                res[rq_index] = answer

            FREQ = 10 * B
            if nb % FREQ == 0:
                print(f'{nb//FREQ}/{len(requests)//FREQ}', end = ' ', flush=True)

        return res
    
if config.seed is None:
    config.seed = 1234 

adapter = EvalHarnessAdapter(batch_size_per_gpu=config.bsz, tokenizer=TokenizerWrapper(pipeline.tokenizer))
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
