from f5_lora.modules.commons import load_model, load_vocoder
from f5_lora.config import Config
from ema_pytorch import EMA
import torch

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float16

config = config

model = load_model(
    device=device,
    config=config,
    dtype=dtype,
    load_pretrained=True,
    ckpt_path=config.train.pretrained_ckpt,
    use_ema=True
)

vocoder = load_vocoder(device)
vocoder.requires_grad_(False)

