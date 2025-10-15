from dataclasses import dataclass, field
from typing import Optional, Literal, Dict


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    n_mel_channels: int = 100
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    mel_spec_type: Literal["vocos", "standard"] = "vocos"
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15


@dataclass
class ModelConfig:
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    ff_mult: int = 2
    text_dim: int = 512
    conv_layers: int = 4


@dataclass
class InferenceConfig:
    ode_method: Literal["euler", "midpoint", "rk4"] = "euler"
    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    fix_duration: Optional[float] = None
    use_ema:bool = True
    ckpt_path:str = None

@dataclass
class TrainConfig:
    pass

@dataclass
class Config:
    mode: Literal["train", "inference"] = "train"
    seed: int = 42
    tokenizer_path:str = 'vocab.txt'
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tempfile_kwargs: Dict[str, bool] = field(default_factory=lambda: {
        "delete_on_close": False
    })


def get_default_config(mode: str = "train") -> Config:
    import sys
    cfg = Config(mode=mode)
    if sys.version_info < (3, 12):
        cfg.tempfile_kwargs = {"delete": False}
    return cfg


# Usage example:
# ---------------
# from config import get_default_config
# cfg = get_default_config("inference")
# print(cfg.model.dim)
# print(cfg.audio.sample_rate)
