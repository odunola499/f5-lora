from torch.optim import AdamW
from bitsandbytes.optim import AdamW as bnb_AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LinearLR, SequentialLR


OPTIMIZER_MAPPING = {
    'bnb': bnb_AdamW,
    'adamw': AdamW,
}

LR_SCHEDULER_MAPPING = {
    'cosine_warmup': get_cosine_schedule_with_warmup,
    'linear_lr': LinearLR,
    'sequential_lr': SequentialLR,
}