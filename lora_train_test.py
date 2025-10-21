from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData
from f5_lora.modules.lora import LoraManager
from datasets import load_dataset

import soundfile as sf

data = load_dataset('ylacombe/expresso')['train']
whisper = data.filter(lambda row: row['style' == 'whisper'])
whisper.push_to_hub('odunola/expresso-whisper')

confusion = data.filter(lambda row: row['style' == 'confusion'])
confusion.push_to_hub('odunola/expresso-confusion')

config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 1
config.train.save_interval = 50
config.train.grad_accumulation_steps=4

train_data= HFData(
    repo_id="ylacombe/expresso",
    name = None,
    split="train",
    text_column="text",
    audio_column="audio",
    stream=False,
)

train_loader = get_loader(config.train.batch_size, config, train_data)
for batch in train_loader:
    print(batch['mel_lengths'])
    break

train_module = TrainModule(config, train_loader)
model = train_module.model
lora_manager = LoraManager(model)

lora_manager.prepare(
    rank=4,
    alpha=8,
    target_modules=None,
    report=True
)
train_module.model = lora_manager.model
print("Initialized LoRA modules.")
print(sum([p.numel() for p in train_module.parameters() if p.requires_grad]))
train_model(config = config, train_module = train_module)