import io
import os
import torch
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, IterableDataset, DataLoader
from f5_lora.modules.utils import MelSpec
from f5_lora.config import HFData, Config
import numpy as np
import librosa
from io import BytesIO
import torchaudio


class StreamHFDataset(IterableDataset):
    def __init__(self,
                 hf_dataset,
                 config: Config,
                 text_column:str = 'text',
                 audio_column:str = 'audio'
                 ):
        super().__init__()
        mel_spec_kwargs = dict(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mel_channels=config.audio.n_mel_channels,
            target_sample_rate=config.audio.sample_rate,
            mel_spec_type=config.audio.mel_spec_type
        )
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sample_rate = config.audio.sample_rate
        self.hop_length = config.audio.hop_length
        self.text_column = text_column
        self.audio_column = audio_column
        hf_dataset = hf_dataset.cast_column(self.audio_column, Audio(decode = False))

        self.config = config
        self.dataset = hf_dataset

    def get_frame_len(self, array:np.ndarray):
        return array / self.hop_length

    def bytes_to_array(self, audio:bytes):
        audio, sr = librosa.load(BytesIO(audio), sr=self.sample_rate)
        return audio, sr

    def normalize(self, tensor:torch.Tensor):
        return (tensor / tensor.abs().max()) * 0.6


    def bytes_to_tensor(self, audio:bytes):
        audio, sr = torchaudio.load(io.BytesIO(audio))
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        audio = audio[:, :int(self.sample_rate * 30.0)]
        duration = audio.shape[-1] / self.sample_rate
        return audio, self.sample_rate, duration

    def process(self, item):
        audio = item[self.audio_column]['bytes']
        text = item[self.text_column]
        tensor, sr, duration = self.bytes_to_tensor(audio)
        tensor = self.normalize(tensor)
        mel_spec = self.mel_spec(tensor).squeeze(0)
        return mel_spec, text


    def __iter__(self):
        for item in self.dataset:
            mel_spec, text = self.process(item)
            yield {
                'mel_spec': mel_spec,
                'text': text
            }

class HFDataset(Dataset):
    def __init__(self,
                 hf_dataset,
                 config: Config,
                 text_column: str = 'text',
                 audio_column: str = 'audio'
                 ):
        super().__init__()
        mel_spec_kwargs = dict(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mel_channels=config.audio.n_mel_channels,
            target_sample_rate=config.audio.sample_rate,
            mel_spec_type=config.audio.mel_spec_type
        )
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.sample_rate = config.audio.sample_rate
        self.hop_length = config.audio.hop_length
        self.text_column = text_column
        self.audio_column = audio_column
        hf_dataset = hf_dataset.cast_column(self.audio_column, Audio(decode=False))

        self.config = config
        self.dataset = hf_dataset

    def get_frame_len(self, array: np.ndarray):
        return array / self.hop_length

    def bytes_to_array(self, audio: bytes):
        audio, sr = librosa.load(BytesIO(audio), sr=self.sample_rate)
        return audio, sr

    def bytes_to_tensor(self, audio: bytes):
        audio, sr = torchaudio.load(io.BytesIO(audio))
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        duration = audio.shape[-1] / self.sample_rate
        return audio, self.sample_rate, duration

    def process(self, item):
        audio = item[self.audio_column]['bytes']
        text = item[self.text_column]
        tensor, sr, duration = self.bytes_to_tensor(audio)
        tensor = self.normalize(tensor)
        mel_spec = self.mel_spec(tensor)
        mel_spec = mel_spec.squeeze(0)
        return mel_spec, text

    def normalize(self, tensor:torch.Tensor):
        return (tensor / tensor.abs().max()) * 0.6

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        mel_spec, text = self.process(self.dataset[item])
        return {
            'mel_spec': mel_spec,
            'text': text
        }

def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])

    transposed_mels = [m.T for m in mel_specs]
    padded = torch.nn.utils.rnn.pad_sequence(transposed_mels, batch_first=True)
    mel_specs = padded.transpose(1,2)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return {
        'mel': mel_specs,
        'mel_lengths': mel_lengths,
        'text': text,
        'text_lengths': text_lengths
    }

def get_loader(
        batch_size,
        config: Config,
        hf_data: HFData
):
    hf_url = hf_data.repo_id
    hf_name = hf_data.name
    hf_split = hf_data.split
    text_column = hf_data.text_column
    audio_column = hf_data.audio_column
    stream = hf_data.stream

    dataset = load_dataset(hf_url, hf_name,split = hf_split, streaming = stream)
    if stream:
        dataset = StreamHFDataset(dataset, config, text_column=text_column, audio_column=audio_column)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle = False,
            drop_last=True,
            num_workers=os.cpu_count(),
            prefetch_factor=2
        )
        print(f"Dataset loaded from {hf_url} and split {hf_split}, streaming mode")
    else:
        dataset = HFDataset(dataset, config, text_column=text_column, audio_column=audio_column)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=os.cpu_count(),
            prefetch_factor=2,
            collate_fn=collate_fn
        )
        print(f"Dataset loaded from {hf_url} and split {hf_split}, number of samples: {len(dataset)}")
    return loader

if __name__ == "__main__":
    hf_url = 'hf-internal-testing/librispeech_asr_demo'
    batch_size = 4
    config = Config()
    hf_split = 'validation'
    loader = get_loader(hf_url, batch_size, config, hf_split=hf_split, stream=True)

    for i, batch in enumerate(loader):
        print(batch['mel'].shape, batch['text'])
        if i > 10:
            break



