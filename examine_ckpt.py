import sys
import math
import torch
from collections import OrderedDict
import re
from safetensors.torch import load_file

if len(sys.argv) != 2:
    print(f"Examines checkpoint keys")
    print("Usage: python examine_ckpt.py in_file")
    exit()

model_path = sys.argv[1]

print("Loading file...")
if model_path.lower().endswith('.safetensors'):
    state_dict = load_file(model_path)
else:
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

state_dict_keys = list(state_dict.keys())
for name in state_dict_keys:
    print(name, state_dict[name].shape, float(state_dict[name].min()), float(state_dict[name].max()))
