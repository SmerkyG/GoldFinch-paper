import sys, os
import os.path
import glob
import torch

if len(sys.argv) != 2:
    print(f"Converts ds moe checkpoint to pth")
    print("Usage: python convert_ds_moe_to_pth.py in_file")
    exit()

ckpt_path_in = sys.argv[1]

state_dict = {}

file_paths = glob.glob(os.path.join(ckpt_path_in, 'checkpoint', '*_model_states.pt'))

for path in file_paths:
    print(f"Processing file {path}")
    file_state_dict = torch.load(path, map_location='cpu', weights_only=False)
    if 'module' in file_state_dict:
        print('getting module from inside state dict')
        file_state_dict = file_state_dict['module']
    keys = list(file_state_dict.keys())
    for k in keys:
        v = file_state_dict[k]
        if not isinstance(v, torch.Tensor):
            del file_state_dict[k]
        else:
            file_state_dict[k] = v.bfloat16()
    state_dict.update(file_state_dict)

ckpt_path_out = os.path.join(ckpt_path_in, 'ckpt.pth')

print(f"Writing file {ckpt_path_out}")

torch.save(state_dict, ckpt_path_out)

print(f"Done!")
