import torch
from safetensors.torch import load_file
from f5_lora.config import Config
from .dit import DIT
from .model import CFM
from vocos import Vocos
from huggingface_hub import hf_hub_download
import os

def load_checkpoint(model,
                    ckpt_path,
                    device: torch.device,
                    dtype=None,
                    use_ema=True):
    device = device.type
    if dtype is None:
        dtype = (
            torch.float16  # todo: try torch.bfloat16
            if "cuda" in device
               and torch.cuda.get_device_properties(device).major >= 7
               and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)
    checkpoint = load_file(ckpt_path, device=device)
    if use_ema:
        checkpoint = {'ema_model_state_dict': checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        response = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    else:
        checkpoint = {"model_state_dict": checkpoint}
        response = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    print(response)

    missing, unexpected = response.missing_keys, response.unexpected_keys
    missing = [i for i in missing if 'extractor' not in i]

    lora_missing = [k for k in missing if "lora" in k.lower()]
    lora_unexpected = [k for k in unexpected if "lora" in k.lower()]

    if any(k for k in missing if "lora" not in k.lower()):
        raise ValueError(f"Missing non-LoRA keys: {missing}")

    if any(k for k in unexpected if "lora" not in k.lower()):
        raise ValueError(f"Unexpected non-LoRA keys: {unexpected}")

    if lora_missing or lora_unexpected:
        print("Found newly initialized LoRA weights. Make sure this is intentional.")

    del checkpoint
    torch.cuda.empty_cache()

    return model


def get_tokenizer(file_path):
    with open(file_path, 'r') as fp:
        vocab_char_map = {}
        for i, char in enumerate(fp):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size


def load_model(
        device: torch.device,
        config: Config,
        use_ema=True,
        dtype=torch.float16,
        load_pretrained=True,
        ckpt_path=None):

    vocab_path = hf_hub_download(
        repo_id=config.hf_repo_id,
        filename='vocab.txt',
        subfolder=config.subfolder

    )
    print('Vocab downloaded to', vocab_path)
    vocab_char_map, vocab_size = get_tokenizer(vocab_path)

    model = CFM(
        transformer=DIT(
            dim=config.model.dim,
            depth=config.model.depth,
            heads=config.model.heads,
            ff_mult=config.model.ff_mult,
            text_dim=config.model.text_dim,
            conv_layers=config.model.conv_layers,
            text_num_embeds=vocab_size,
            mel_dim=config.audio.n_mel_channels
        ),
        mel_spec_kwargs=dict(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mel_channels=config.audio.n_mel_channels,
            target_sample_rate=config.audio.sample_rate,
            mel_spec_type=config.audio.mel_spec_type
        ),
        odeint_kwargs=dict(
            method=config.inference.ode_method
        ),
        vocab_char_map=vocab_char_map
    ).to(device)

    print('Loading checkpoint')

    if load_pretrained:
        if ckpt_path:
            print('Loading model checkpoint from', ckpt_path)
        elif os.path.exists(config.ckpt_path) and config.ckpt_path is not None:
            ckpt_path = config.ckpt_path
            print('Loading model checkpoint from', ckpt_path)
        else:
            print('ckpt_path not found, downloading from HF hub')
            assert config.hf_repo_id is not None, "hf_repo_id must be specified in config"
            assert config.filename is not None, "filename must be specified in config"
            ckpt_path = hf_hub_download(
                repo_id= config.hf_repo_id,
                filename=config.filename,
                subfolder=config.subfolder
            )
            print('Checkpoint downloaded to', ckpt_path)
        model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params / 1e6:.2f}M")
    print('Loaded model')
    return model


def load_vocoder(device, hf_cache_dir=None):
    repo_id = "charactr/vocos-mel-24khz"
    config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
    model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")

    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    from vocos.feature_extractors import EncodecFeatures

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder




