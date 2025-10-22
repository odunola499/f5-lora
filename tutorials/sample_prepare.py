import torch
from f5_lora.config import Config
from f5_lora.modules.commons import load_model, load_vocoder
from f5_lora.modules.lora import LoraManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

config = Config()
model = load_model(
                device=device,
                config=config,
                dtype=dtype,
                load_pretrained=False,
                ckpt_path=None,
                use_ema=True
)
rank = 8
alpha = 8

lora_manager = LoraManager(model)
lora_manager.prepare(
    rank=rank,
    alpha=alpha,
    target_modules=None,
    report=True
)