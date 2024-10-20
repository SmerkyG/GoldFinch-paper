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

for name, p in state_dict.items():
    if p.numel() == 0:
        print(name, p.dtype, p.shape)
    else:
        print(name, p.dtype, p.shape, float(p.min()), float(p.max()))
