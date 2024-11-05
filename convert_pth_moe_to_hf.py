import sys, os, os.path
import torch
from transformers.modeling_utils import convert_file_size_to_int, dtype_byte_size
from safetensors.torch import save_file as safe_save_file
import json
from typing import Dict, Union, Any

if len(sys.argv) != 3:
    print(f"Converts pth moe checkpoint to huggingface")
    print("Usage: python convert_pth_moe_to_hf.py in_file out_file")
    exit()

model_path = sys.argv[1]
save_path = sys.argv[2]

print("Loading file...")
state_dict = torch.load(model_path, map_location='cpu')

state_dict['head.weight'] = state_dict.pop('model.head.weight')
state_dict['model.embeddings.weight'] = state_dict.pop('model.emb.weight')    

print("Converting entries...")

state_dict_keys = list(state_dict.keys())
for original_name in state_dict_keys:
    name = original_name
    name = name.replace('.ln0.','.pre_ln.')
    name = name.replace('.moe.deepspeed_moe.experts.deepspeed_experts.','.experts.')
    name = name.replace('.ffn_key.','.key.')
    name = name.replace('.ffn_value.','.value.')
    name = name.replace('.att.','.attention.')
    name = name.replace('.ffn.','.feed_forward.')
    name = name.replace('.feed_forward.key.','.feed_forward.shared_expert.key.')
    name = name.replace('.feed_forward.value.','.feed_forward.shared_expert.value.')

    if name != original_name:
        state_dict[name] = state_dict.pop(original_name)

state_dict = dict(sorted(state_dict.items()))

print("Writing files...")

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = "pytorch_model.bin"
):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size.

    The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so there is no
    optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For example, if the
    limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as [6GB], [6+2GB],
    [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger than `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dictionary of a model to save.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            The name of the model save file.
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)

    sharded_state_dicts = [{}]
    last_block_size = 0
    total_size = 0
    #storage_id_to_block = {}

    for key, weight in state_dict.items():
        # when bnb serialization is used the weights in the state dict can be strings
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(weight, str):
            continue
        #else:
        #    storage_id = id_tensor_storage(weight)

        # # If a `weight` shares the same underlying storage as another tensor, we put `weight` in the same `block`
        # if storage_id in storage_id_to_block:
        #     block_id = storage_id_to_block[storage_id]
        #     sharded_state_dicts[block_id][key] = weight
        #     continue

        weight_size = weight.numel() * dtype_byte_size(weight.dtype)

        # If this weight is going to tip up over the maximal size, we split, but only if we have put at least one
        # weight in the current shard.
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0

        sharded_state_dicts[-1][key] = weight
        last_block_size += weight_size
        total_size += weight_size
        #storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # If we only have one shard, we return it
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # Otherwise, let's build the index
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # Add the metadata
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index

shards, index = shard_checkpoint(state_dict, '5GB', 'model.safetensors')
for shard_file, shard in shards.items():
    safe_save_file(shard, os.path.join(save_path, shard_file), metadata={"format": "pt"})

if index is not None:
    with open(os.path.join(save_path, 'model.safetensors.index.json'), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

#torch.save(state_dict,sys.argv[2])
print("DONE. Files written.")
