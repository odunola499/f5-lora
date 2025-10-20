from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData

config = Config()
config.train.learning_rate = 1e-4
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 2
config.train.grad_accumulation_steps=4

train_data= HFData(
    repo_id="vbrydik/eng-male-speaker-0-v1",
    name = None,
    split="train",
    text_column="text",
    audio_column="audio",
    stream=True,
)

train_loader = get_loader(config.train.batch_size, config, train_data)
for batch in train_loader:
    print(batch['mel_lengths'])
    break
train_module = TrainModule(config, train_loader)
train_model(config = config, train_module = train_module)