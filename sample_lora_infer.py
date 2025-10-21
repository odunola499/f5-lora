import torch
from f5_lora.infer.inference import Inference, Config
from f5_lora.modules.lora import LoraManager
from f5_lora.modules.commons import load_model, load_vocoder
import soundfile as sf

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adapter_path = "path/to/your/checkpoint.safetensors"

model = load_model(
    device=device,
    use_ema=config.inference.use_ema,
    config=config
).eval()

vocoder = load_vocoder(device = device).eval()

manager = LoraManager(model)
manager.load(adapter_path, name = 'whisper')

infer = Inference(config)

ref_text = "Monday, there's gonna be haze, but Tuesday, look for thunderstorms."
ref_audio = 'sample.wav'
gen_text = 'Can I get a quick shoutout to all friends that love good food and making money? If you are one of them, then you are at the right place.'

audio_segment, final_sample_rate, spectrogram = infer(ref_audio=ref_audio, ref_text=ref_text,
                                                                        gen_text=gen_text)
sf.write('output.wav', audio_segment, final_sample_rate)