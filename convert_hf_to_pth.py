import sys
import torch
import transformers

if len(sys.argv) != 3:
    print("Usage:\n\tpython convert_hf_to_pth.py HF_PATH_IN PATH_OUT")
    exit()

_, path_in, path_out = sys.argv

torch.set_default_dtype(torch.bfloat16)

print(f"Loading {path_in}...")
model = transformers.AutoModelForCausalLM.from_pretrained(path_in)
print(f"Saving {path_out}...")
torch.save(model.state_dict(), path_out)
print(f"Done!")
