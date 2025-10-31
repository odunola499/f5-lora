# F5-LoRA: Efficient Low-Rank Adaptation for Flow-Matching TTS

This repository extends the F5-TTS (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching) model repo [code](https://github.com/SWivid/F5-TTS) with easy support for LoRA and full parameter finetuning. This allows for easy lightweight style transfer and voice cloning.

You should be able to make a clone of your voice with remarkable accuracy after finetune on a google colab, or a voice style + acceptable zero shot voice cloning.


Training rewritten in Pytorch Lightning.
Check out the sample files in the tutorial/ repo for quckstart examples.


[▶️ Before Finetune ](./samples/before_finetune.wav)

[▶️ After Finetune](./samples/after_finetune.wav)
## Features

- **LoRA Fine-tuning**: Efficient adaptation using Low-Rank matrices
- **Full Parameter Fine-tuning**: Traditional full model fine-tuning support
- **Multiple Adapter Management**: Load, save, and swap between different LoRA adapters
- **HuggingFace Integration**: Direct dataset loading from HuggingFace Hub

## Installation

```bash
git clone https://github.com/odunola499/f5-tts.git
cd f5-tts
pip install -e .
```

## Quick Start

### LoRA Fine-tuning

LoRA (Low-Rank Adaptation) allows you to fine-tune the model with a fraction of the parameters. You could have multiple adapters for different styles or speakers.
#### Training a LoRA Adapter

```python
from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData
from datasets import load_dataset


config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 1
config.train.save_interval = 50
config.train.grad_accumulation_steps = 4


train_data = HFData(
    repo_id='ylacombe/expresso',
    name=None,
    split="train",
    text_column="text",
    audio_column="audio",
    stream=False,
)

train_loader = get_loader(config.train.batch_size, config, train_data)


train_module = TrainModule(config, train_loader, lora=True, alpha = 32, rank = 128)

print("Initialized LoRA modules.")


train_model(config=config, train_module=train_module)
```
or if you have a manifest file that looks like this

```commandline
{"audio": "path/to/audio_1.wav", "text": "Hello, this is the first sample."}
{"audio": "path/to/audio_2.wav", "text": "This is another recording of the same speaker."}
{"audio": "path/to/audio_3.wav", "text": "The quick brown fox jumps over the lazy dog."}
```

you can run training like so
 
```commandline
from f5_lora.train import get_local_loader, TrainModule, train_model
from f5_lora.config import Config, LocalData

config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 2
config.train.save_interval = 50
config.train.grad_accumulation_steps = 4

local_data = LocalData(
    audio_dir="data/audio",
    manifest_file="data/manifest.jsonl",
    audio_column="audio",
    text_column="text"
)

train_loader = get_local_loader(batch_size=config.train.batch_size, config=config, local_data=local_data)

train_module = TrainModule(config, train_loader, lora=True)

train_model(config=config, train_module=train_module)


```
This would also apply for full parameter finetune, below.
#### LoRA Inference

```python
import torch
from f5_lora.infer.inference import Inference, Config
from f5_lora.modules.lora import LoraManager
from f5_lora.modules.commons import load_model, load_vocoder
import soundfile as sf


config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = load_model(
    device=device,
    use_ema=config.inference.use_ema,
    config=config
).eval()


vocoder = load_vocoder(device=device).eval()


adapter_path = "adapter_path.safetensors"
manager = LoraManager(model)
manager.load(adapter_path, name='my_adapter')


infer = Inference(config)


ref_text = "Monday, there's gonna be haze, but Tuesday, look for thunderstorms."
ref_audio = 'reference_sample.wav'
gen_text = 'Can I get a quick shoutout to all friends that love good food and making money?'

audio_segment, final_sample_rate, spectrogram = infer(
    ref_audio=ref_audio, 
    ref_text=ref_text,
    gen_text=gen_text
)


sf.write('output.wav', audio_segment, final_sample_rate)
```

### Full Parameter Fine-tuning

For scenarios where you need maximum adaptation capability and performance:

```python
from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData


config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 4
config.train.save_interval = 50
config.train.grad_accumulation_steps = 4


train_data = HFData(
    repo_id="ylacombe/expresso",
    name=None,
    split="train",
    text_column="text",
    audio_column="audio",
    stream=False,
)


train_loader = get_loader(config.train.batch_size, config, train_data)
train_module = TrainModule(config, train_loader)  # No LoRA flag = full fine-tuning


train_model(config=config, train_module=train_module)
```
Check the `tutorial\` repo for steps on full parameter inference

## Lora Manager

The `LoraManager` class allows ease of use for managing multiple adapters:

### Basic LoRA Operations

```python
from f5_lora.modules.lora import LoraManager

# Initialize manager
manager = LoraManager(model)

# Prepare LoRA layers
manager.prepare(rank=4, alpha=8, target_modules=["to_q", "to_v", "proj_out"])

# Save adapter
manager.save("my_adapter.safetensors")

# Load adapter
manager.load("my_adapter.safetensors", name="style1")

# Reset (remove LoRA) to base model
manager.reset()
```

### Multiple Adapter Management

```python
# Load multiple adapters
manager.load("whisper_style.safetensors", name="whisper")
manager.load("excited_style.safetensors", name="excited")
manager.load("calm_style.safetensors", name="calm")

# Swap between adapters during inference
manager.swap("whisper")  # Switch to whisper style

manager.swap("excited")  # Switch to excited style

# Delete an adapter
manager.delete("calm")
```
### Notes
- I set `alpha` to 32 and `rank` to 64 as default values as this showed the best performance in my tests, but these can be adjusted based on your requirements.
- Do tinker with the LoRA target modules as well. Currently all Linear layers in the model are specified as target modules. check `modules/lora.py`. You may get much better results.
- Much better performance with full parameter finetune but please try out different hyperparameters for LoRA.
- When training, for convenience two different checkpoints types are saved. The folder `checkpoints`(by default) saved the ema weights in .safetensors. This makes it easy for direct inference with the sample code given. the `train_checkpoints/` folder saves the entire trainer state and can be used to resume a training run. 

## Acknowledgments

- Original F5-TTS implementation seen [here](https://github.com/SWivid/F5-TTS)
- LoRA paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- HuggingFace for their opensource framework.
