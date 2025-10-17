from dataclasses import dataclass, field
from typing import Optional, Literal, Dict
from pydantic import BaseModel

class HFData(BaseModel):
    repo_id:str
    name:str
    split:str
    text_column:str = 'text'
    audio_column:str = 'audio'
    stream:bool = True

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

@dataclass
class TrainConfig:
    epochs:Optional[int]
    learning_rate:float
    max_steps:Optional[int] = 20000
    warmup_steps:int = 2000
    keep_last_n_checkpoints:int = 2
    ckpt_path:str = 'checkpoints'
    pretrained_ckpt:Optional[str] = None
    resume_run:bool = True
    batch_size:int = 8
    grad_accumulation_steps:Optional[int] = 2
    max_grad_norm:int = 1.0
    noise_scheduler:Optional[str] = None
    log_to:Literal['wandb','csv'] = 'wandb'
    wandb_project:str = 'F5_TTS'
    wandb_run_name:Optional[str] = None
    log_samples:bool = True
    optimizer:Literal['bnb', 'adamw'] = 'bnb'
    lr_scheduler:Literal['cosine', 'linear_lr', 'sequential_lr'] = 'cosine'
    save_interval:Optional[int] = 1000
    val_interval:Optional[int] = 1000



@dataclass
class Config:
    mode: Literal["train", "inference"] = "train"
    seed: int = 42
    ckpt_path:Optional[str] = None
    tokenizer_path:str = 'vocab.txt'
    hf_repo_id: str = 'SWivid/F5-TTS'
    filename: str = 'model_1250000.safetensors'
    subfolder: str = 'F5TTS_v1_Base'
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

