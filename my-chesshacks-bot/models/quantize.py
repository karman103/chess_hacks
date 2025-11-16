import torch

state = torch.load("policy_value_sf.pt", map_location="cpu")
for k in state:
    state[k] = state[k].half()   # convert to FP16

torch.save(state, "policy_value_sf_fp16.pt")
