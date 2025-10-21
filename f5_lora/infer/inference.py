import torch
from f5_lora.modules.commons import load_model, load_vocoder
from f5_lora.config import Config
from pydub import AudioSegment, silence
import numpy as np
import tempfile
import soundfile as sf
import hashlib
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import torchaudio
from f5_lora.modules.utils import chunk_text
from torch.profiler import profile, ProfilerActivity, record_function


class Inference:
    def __init__(self, config:Config, profile_model = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(
            device = self.device,
            use_ema=config.inference.use_ema,
            config = config
                        ).eval()

        self.vocoder = load_vocoder(device = self.device).eval()
        self.config = config
        self.profile_model = profile_model

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

    def preprocess_ref_audio_text(self, ref_audio_orig, ref_text, show_info=print):
        print("Converting audio...")

        # Compute a hash of the reference audio file
        with open(ref_audio_orig, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_hash = hashlib.md5(audio_data).hexdigest()

        global _ref_audio_cache

        if audio_hash in _ref_audio_cache:
            print("Using cached preprocessed reference audio...")
            ref_audio = _ref_audio_cache[audio_hash]

        else:  # first pass, do preprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", **self.config.tempfile_kwargs) as f:
                temp_path = f.name

            aseg = AudioSegment.from_file(ref_audio_orig)

            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        show_info("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 12s, clipping short. (3)")

            aseg = self.remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
            aseg.export(temp_path, format="wav")
            ref_audio = temp_path

            # Cache the processed reference audio
            _ref_audio_cache[audio_hash] = ref_audio

    def infer_batch_process(self,
                            ref_audio,
                            ref_text,
                            gen_text_batches,
                            streaming=False,
                            chunk_size=2048,
                            target_rms=0.1,
                            target_sample_rate=24000,
                            speed=1.0,
                            hop_length=256,
                            ):
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(self.device)

        generated_waves = []
        spectrograms = []

        if len(ref_text[-1].encode("utf-8")) == 1:
            ref_text = ref_text + " "

        def process_batch(gen_text):
            local_speed = speed
            if len(gen_text.encode("utf-8")) < 10:
                local_speed = 0.3

            # Prepare the text
            text_list = [ref_text + gen_text]
            final_text_list = text_list
            #final_text_list = convert_char_to_pinyin(text_list)

            ref_audio_len = audio.shape[-1] // hop_length
                # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

            # inference
            with torch.inference_mode():
                generated, _ = self.model.sample(
                    cond=audio,
                    text=final_text_list,
                    duration=duration,
                    steps=self.config.inference.nfe_step,
                    cfg_strength=self.config.inference.cfg_strength,
                    sway_sampling_coef=self.config.inference.sway_sampling_coef,
                )
                del _

                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len:, :]
                generated = generated.permute(0, 2, 1)
                generated_wave = self.vocoder.decode(generated)
                if rms < target_rms:
                    generated_wave = generated_wave * rms / target_rms

                # wav -> numpy
                generated_wave = generated_wave.squeeze().cpu().numpy()

                if streaming:
                    for j in range(0, len(generated_wave), chunk_size):
                        yield generated_wave[j: j + chunk_size], target_sample_rate
                else:
                    generated_cpu = generated[0].cpu().numpy()
                    del generated
                    yield generated_wave, generated_cpu

        if streaming:
            for gen_text in tqdm(gen_text_batches):
                for chunk in process_batch(gen_text):
                    yield chunk
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
                for future in tqdm(futures):
                    result = future.result()
                    if result:
                        generated_wave, generated_mel_spec = next(result)
                        generated_waves.append(generated_wave)
                        spectrograms.append(generated_mel_spec)

            cross_fade_duration = self.config.audio.cross_fade_duration
            if generated_waves:
                if cross_fade_duration <= 0:
                    # Simply concatenate
                    final_wave = np.concatenate(generated_waves)
                else:
                    # Combine all generated waves with cross-fading
                    final_wave = generated_waves[0]
                    for i in range(1, len(generated_waves)):
                        prev_wave = final_wave
                        next_wave = generated_waves[i]

                        # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                        cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                        cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                        if cross_fade_samples <= 0:
                            # No overlap possible, concatenate
                            final_wave = np.concatenate([prev_wave, next_wave])
                            continue

                        # Overlapping parts
                        prev_overlap = prev_wave[-cross_fade_samples:]
                        next_overlap = next_wave[:cross_fade_samples]

                        # Fade out and fade in
                        fade_out = np.linspace(1, 0, cross_fade_samples)
                        fade_in = np.linspace(0, 1, cross_fade_samples)

                        # Cross-faded overlap
                        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                        # Combine
                        new_wave = np.concatenate(
                            [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                        )

                        final_wave = new_wave

                # Create a combined spectrogram
                combined_spectrogram = np.concatenate(spectrograms, axis=1)

                yield final_wave, target_sample_rate, combined_spectrogram

            else:
                yield None, target_sample_rate, None

    def __call__(self,
                 ref_audio,
                 ref_text,
                 gen_text):
        audio, sr = torchaudio.load(ref_audio)
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * self.config.inference.speed)
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
    config.ckpt_path = '/root/f5-tts/checkpoints/model_400.safetensors'
    infer = Inference(config)
    print('Loaded')

    ref_text = "Monday, there's gonna be haze, but Tuesday, look for thunderstorms."
    ref_audio = 'sample.wav'
    gen_text = 'Can I get a quick shoutout to all friends that love good food and making money? If you are one of them, then you are at the right place.'
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        audio_segment, final_sample_rate, spectrogram = infer(ref_audio=ref_audio, ref_text=ref_text,
                                                                        gen_text=gen_text)
    sf.write('output.wav', audio_segment, final_sample_rate)
    #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))