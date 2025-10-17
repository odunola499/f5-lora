import math

import torch
from torch import nn

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r = 8, alpha = 16, bias = True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a = math.sqrt(5))
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        else:
            self.lora_A = self.lora_B = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_update = torch.matmul(self.lora_B, self.lora_A) * self.scale
            result += torch.nn.functional.linear(x, lora_update)
        return result

