import math
import torch
from torch import nn
from safetensors.torch import save_file, safe_open

class LoraLinear(nn.Module):
    def __init__(self, layer:nn.Linear, r = 4, alpha = 8):
        super().__init__()
        self.base = layer
        self.scale = alpha / r

        self.lora_A = nn.Parameter(torch.randn(r, layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(layer.out_features, r))
        self.base.requires_grad_(False)
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def forward(self, x):
        output = self.base(x)
        delta = (self.lora_B @ self.lora_A) * self.scale
        lora_out = self.nn.functional.linear(x, delta)
        output = output + lora_out
        return output

    def get_lora_state_dict(self):
        return {k:v for k,v in self.state_dict().items() if 'lora' in k.lower()}

    def load_lora_state_dict(self, state_dict):
        response = self.load_state_dict(state_dict, strict = False)
        assert 'lora' not in response.missing_keys, f'Missing keys in LoRA state dict: {response.missing_keys}'
        print('LoRA state dict loaded successfully.')

modules = [
    'pwconv1',
    'pwconv2',
    'time_mlp',
    'proj',
    'proj_out',
    'to_q',
    'to_k',
    'to_v',
    'ff.2'
    'to_out.0'
]

class LoraManager:
    def __init__(self, model:nn.Module):
        self.model = model
        self.model.requires_grad_(False)
        self.alpha = None
        self.rank = None

    def prepare_model(self, rank = 4, alpha = 8, target_modules=None, report = True):
        for name, module in self.model.named_modules():
            if any(target_module in name for target_module in target_modules) and isinstance(module, nn.Linear):
                lora_linear = LoraLinear(module, rank, alpha)
                self._set_submodule(self.model, name, lora_linear)
        if report:
            total_params = sum(p.numel() for p in self.model.parameters())
            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"Total parameters: {total_params}, LoRA parameters: {lora_params}, Trainable percentage: {lora_params/total_params*100:.2f}%"
            )
        self.alpha = 8
        self.rank = 4

    def clean_model(self):
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLinear):
                linear_layer = module.base
                self._set_submodule(self.model, name, linear_layer)
        print("Loaded Original Linear Layers")
        self.alpha = None
        self.rank = None

    def save_adapter(self, path):
        assert '.safetensors' in path, "adapter checkpoints are valid .safetensors files"
        assert self.alpha is not None, "LoRA layers are not available in model."
        state_dict = {
            k:v for k,v in self.model.state_dict().items() if 'lora' in k.lower()
        }
        weights =  {'state_dict': state_dict,
                    'alpha': torch.tensor(self.alpha, dtype = torch.long),
                    'rank': torch.tensor(self.rank, dtype = torch.long)
                 }
        save_file(weights, path)

    def load_adapter(self, path):
        with safe_open(path, framework="pt") as f:
            rank = f.get_tensor('rank').item()
            alpha = f.get_tensor('alpha').item()
            state_dict = f.get_tensor('state_dict')

        old_alpha, old_rank = self.alpha, self.rank
        if old_alpha is not None:
            if old_alpha == alpha and old_rank == rank:
                self.model.load_state_dict(state_dict, strict = False)
            else:
                self.clean_model()
                self.prepare_model(rank = rank, alpha = alpha, target_modules = modules)
                self.model.load_state_dict(state_dict, strict = False)
        else:
            self.prepare_model(rank=rank, alpha=alpha, target_modules=modules)
            self.model.load_state_dict(state_dict, strict=False)


    def _set_submodule(self, model, name, new_module):
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            elif isinstance(parent, (nn.Sequential, nn.ModuleList)) and part.isdigit():
                parent = parent[int(part)]
            else:
                raise AttributeError(f"Module '{parent}' has no attribute '{part}'")
        child = parts[-1]

        if hasattr(parent, child):
            setattr(parent, child, new_module)
        elif isinstance(parent, (nn.Sequential, nn.ModuleList)) and child.isdigit():
            parent[int(child)] = new_module
        else:
            raise KeyError(f"Cannot set submodule: {name}")



