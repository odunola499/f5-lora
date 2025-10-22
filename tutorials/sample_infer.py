from f5_lora.infer.inference import Inference, Config

import soundfile as sf

config = Config()
#config.ckpt_path = "path/to/your/checkpoint.safetensors"

infer = Inference(config)

ref_text = "Monday, there's gonna be haze, but Tuesday, look for thunderstorms."
ref_audio = 'sample.wav'
gen_text = 'Can I get a quick shoutout to all friends that love good food and making money? If you are one of them, then you are at the right place.'

audio_segment, final_sample_rate, spectrogram = infer(ref_audio=ref_audio, ref_text=ref_text,
                                                                        gen_text=gen_text)
sf.write('output.wav', audio_segment, final_sample_rate)