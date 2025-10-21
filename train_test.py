from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData
from datasets import load_dataset
import time

import soundfile as sf
#data = load_dataset('ylacombe/expresso')['train']
#sf.write('sample.wav', data[2]['audio']['array'], data[2]['audio']['sampling_rate'])
#print(data[2]['text'])
#time.sleep(5)


config = Config()
config.train.learning_rate = 1e-5
config.train.max_steps = 3000
config.train.warmup_steps = 300
config.train.batch_size = 4
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
train_model(config = config, train_module = train_module)