import copy
import math
import torch
from torch.nn import functional as F
import copy
from torch import nn
from safetensors.torch import save_file, safe_open

class LoraLinear(nn.Module):
    def __init__(self, layer:nn.Linear, r = 4, alpha = 8):
        super().__init__()
        self.base = copy.deepcopy(layer)
        self.scale = alpha / r

        dtype = layer.weight.dtype
        device = layer.weight.device

        self.lora_A = nn.Parameter(torch.randn(r, layer.in_features, dtype = dtype, device = device))
        self.lora_B = nn.Parameter(torch.zeros(layer.out_features, r, dtype = dtype, device = device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data *= 0.01

        self.base.requires_grad_(False)

    def forward(self, x):
        output = self.base(x)
        delta = (self.lora_B @ self.lora_A) * self.scale
        delta = delta.to(dtype = x.dtype, device = x.device)
        lora_out = torch.nn.functional.linear(x, delta)
        return output + lora_out

    def lora_state(self):
        return {k: v for k, v in self.state_dict().items() if "lora" in k}

modules = [
    'pwconv1',
    'pwconv2',
    'proj',
    'proj_out',
    'to_q',
    'to_k',
    'to_v',
    'ff.2',
    '0.0'
    'to_out.0'
]

def _set_submodule(model, name, new_module):
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part) if hasattr(parent, part) else parent[int(part)]
    last = parts[-1]
    if hasattr(parent, last):
        setattr(parent, last, new_module)
    else:
        parent[int(last)] = new_module


class LoraManager:
    def __init__(self, model:nn.Module):
        self.model = model
        self.model.requires_grad_(False)
        self.alpha = None
        self.rank = None
        self.adapters = {}
        self.active = None
        self.MODULES = modules

    def prepare(self, rank = 4, alpha = 8, report = True):
        self.rank = rank
        self.alpha = alpha

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print(f'Applying LoRA to module: {name} | {module}')
                lora_module = LoraLinear(module, rank, alpha)
                _set_submodule(self.model, name,lora_module)
        if report or self.model.training:
            total = sum([p.numel() for p in self.model.parameters()])
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"LoRA params: {trainable}/{total} ({trainable / total * 100:.3f}%)")

    def reset(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLinear):
                _set_submodule(self.model, name, module.base)
        self.alpha = self.rank = None
        print('LoRA adapters removed.')


    def save(self, path):
        assert self.alpha is not None, "LoRA adapters are not active."
        state_dict = {f'{k}':v for k,v in self.model.state_dict().items() if 'lora' in k}
        state_dict['alpha'] = torch.tensor(self.alpha)
        state_dict['rank'] = torch.tensor(self.rank)
        save_file(state_dict, path)

    def load(self, path, name = None, activate = True):
        with safe_open(path, framework='pt', device = 'cpu') as fp:
            state_dict = {k: fp.get_tensor(k) for k in fp.keys()}
            alpha = state_dict.pop('alpha').item()
            rank = state_dict.pop('rank').item()

        if self.alpha != alpha or self.rank != rank:
            self.prepare(rank = rank, alpha = alpha)

        state_dict = {i.replace('ema_model.',''):j for i,j in state_dict.items()}
        self.model.load_state_dict(state_dict, strict = False)
        if name:
            self.adapters[name] = state_dict
            if activate:
                self.active = name
                print(f'Activated LoRA adapter: {name}')
        print(f'Loaded LoRA adapter from {path}.')

    def swap(self, name):
        assert name in self.adapters, f'No LoRA adapter named {name} found.'
        state_dict = self.adapters[name]
        self.model.load_state_dict(state_dict, strict = False)
        self.active = name
        print(f'Swapped to LoRA adapter: {name}')

    def delete(self, name):
        if name in self.adapters:
            del self.adapters[name]
            self.active = None
            print(f'Deleted LoRA adapter: {name}')
        else:
            print(f'No LoRA adapter named {name} found.')


