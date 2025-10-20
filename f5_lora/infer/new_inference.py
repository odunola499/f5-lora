import torch
from f5_lora.modules.commons import load_model, load_vocoder
from f5_lora.config import Config
from pydub import AudioSegment, silence
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
import torchaudio
from f5_lora.modules.utils import chunk_text
from torch.profiler import profile, ProfilerActivity



class Inference:
    def __init__(self, config:Config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(
            device=self.device,
            use_ema=config.inference.use_ema,
            config=config
        ).eval()

        self.vocoder = load_vocoder(device=self.device).eval()
        self.config = config

    def remove_silence_edges(self, audio:AudioSegment, silence_threshold = -42):
        non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold = silence_threshold)
        audio = audio[non_silent_start_idx:]

        non_silent_end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS < silence_threshold:
                non_silent_end_duration -= 1
            non_silent_end_duration -= 0.001

        trimmed_audio = audio[: int(non_silent_end_duration * 1000)]
        return trimmed_audio

    def preprocess_ref_audio_text(self, ref_audio_path:str, max_duration = 12.0):
        info = torchaudio.info(ref_audio_path)
        duration = info.num_frames / info.sample_rate
        if duration > max_duration:
            raise ValueError(f"Reference audio duration {duration} exceeds maximum allowed {max_duration} seconds.")
        return ref_audio_path

    def process_batch(self, gen_text, ref_text, audio, speed, rms, target_rms, streaming = False, chunk_size = 2048):
        local_speed = speed

        ref_text_len = len(ref_text.encode('utf-8'))
        gen_text_len = len(gen_text.encode('utf-8'))
        if gen_text_len < 10:
            local_speed = 0.3

        text_list = [ref_text + gen_text]
        final_text_list = text_list

        ref_audio_len = audio.shape[-1] // self.config.audio.hop_length
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        with torch.inference_mode():
            generated, _ = self.model.sample(
                cond = audio,
                text = final_text_list,
                duration = duration,
                steps = self.config.inference.nfe_step,
                cfg_strength = self.config.inference.cfg_strength,
                sway_sampling_coef = self.config.inference.sway_sampling_coef,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            generated_wave = self.vocoder.decode(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            generated_wave = generated_wave.squeeze().cpu().numpy()
            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j: j + chunk_size], self.config.audio.sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu



    def infer_batch_process(self,
                            ref_audio,
                            ref_text,
                            gen_text_batches,
                            streaming = False,
                            chunk_size = 2048,
                            target_rms = 0.1,
                            target_sample_rate = 24000,
                            speed = 1.0,
                            hop_length = 256):
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(self.device)

        if not ref_text.endswith(" "):
            ref_text += " "

        if streaming:
            for gen_text in tqdm(gen_text_batches):
                for chunk in self.process_batch(
                    gen_text,ref_text, audio, speed, rms, target_rms, streaming = True, chunk_size = chunk_size
                ):
                    yield chunk
        else:
            generated_waves, spectrograms = [], []
            for gen_text in gen_text_batches:
                generated_wav, mel_spec = next(self.process_batch(
                    gen_text,ref_text, audio, speed, rms, target_rms, streaming = False, chunk_size = 2048
                ))
                generated_waves.append(generated_wav)
                spectrograms.append(mel_spec)

            cross_fade_duration = self.config.audio.cross_fade_duration
            if cross_fade_duration <= 0:
                final_wave = np.concatenate(generated_waves, axis=0)
            else:
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave =  final_wave
                    next_wave = generated_waves[i]
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            combined_spectrogram = np.concatenate(spectrograms, axis=0)
            yield final_wave, target_sample_rate, combined_spectrogram

    def __call__(self,
                 ref_audio, ref_text, gen_text):
        audio, sr = torchaudio.load(ref_audio)

        silence = torch.zeros(audio.shape[0], int(50e-3 * sr))
        audio = torch.cat([audio, silence], dim=-1)
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (
                    22 - audio.shape[-1] / sr) * self.config.inference.speed)
        gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

        audio_segment, final_sample_rate, spectrogram = next(
            self.infer_batch_process(
                (audio, sr),
                ref_text,
                gen_text_batches
            )
        )
        return audio_segment, final_sample_rate, spectrogram


if __name__ == "__main__":
    config = Config()
    infer = Inference(config)
    print('Loaded')

    ref_text = 'A meaningful livelihood.'
    ref_audio = 'reference.wav'
    gen_text = 'Today is a good day.'
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        audio_segment, final_sample_rate, spectrogram = infer(ref_audio=ref_audio, ref_text=ref_text,
                                                                        gen_text=gen_text)
    sf.write('output.wav', audio_segment, final_sample_rate)
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))




