from f5_lora.train import get_loader, TrainModule, train_model
from f5_lora.config import Config, HFData

config = Config()
train_data= HFData(
    repo_id="hf-internal-testing/librispeech_asr_dummy",
    split="train",
    text_column="text",
    audio_column="audio",
    stream=True,
)

train_loader = get_loader(config.train.batch_size, config, train_data)
train_module = TrainModule(config, train_loader)
train_model(config = config, train_module = train_module)