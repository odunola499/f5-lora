import torch
from f5_lora.infer.inference import Inference, Config
from f5_lora.modules.lora import LoraManager
from f5_lora.modules.commons import load_model, load_vocoder
import soundfile as sf

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adapter_path = "path_to_adapter.safetensors"

model = load_model(
    device=device,
    use_ema=config.inference.use_ema,
    config=config
).eval()

vocoder = load_vocoder(device = device).eval()

manager = LoraManager(model)
manager.load(adapter_path, name = 'audio')

infer = Inference(config,model = manager.model.to(device = device, dtype = torch.float32))

ref_text = "Why are you beating up my jukebox?"
ref_audio = 'audio.wav'
gen_text = 'What are we going to figure out today?'

audio_segment, final_sample_rate, spectrogram = infer(ref_audio=ref_audio, ref_text=ref_text,
                                                                        gen_text=gen_text)
sf.write('output.wav', audio_segment, final_sample_rate)
print('done')