from dit import DIT
from model import CFM
import torch
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos
import re
import numpy as np
import hashlib, tempfile, sys
from concurrent.futures import ThreadPoolExecutor
import torchaudio
import soundfile as sf


tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

config = {
        'dim':1024,
        'depth':22,
        'heads':16,
        'ff_mult':2,
        'text_dim':512,
        'conv_layers':4
    }




def chunk_text(text, max_chars = 135):
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_checkpoint(model, ckpt_path, device: torch.device, dtype=None, use_ema=True):
    device = device.type
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    ckpt_path,
    device,
    mel_spec_type=mel_spec_type,
    vocab_file='vocab.txt',
    ode_method=ode_method,
    use_ema=True,

):
    model_cls = DIT
    model_cfg = config

    vocab_char_map, vocab_size = get_tokenizer(vocab_file)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params / 1e6:.2f}M")
    print('Loaded model')

    return model

def get_tokenizer(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_char_map = {}
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    vocab_size = len(vocab_char_map)
    return vocab_char_map, vocab_size

def load_vocoder(device, hf_cache_dir=None):
    repo_id = "charactr/vocos-mel-24khz"
    config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
    model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")


    vocoder = Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    from vocos.feature_extractors import EncodecFeatures

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    return vocoder



class Inference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model('model_1250000.safetensors', device = self.device)
        self.model = model.eval()
        self.vocoder = load_vocoder(device = self.device).eval()


    def remove_silence_edges(self, audio, silence_threshold = -42):
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
        show_info("Converting audio...")

        # Compute a hash of the reference audio file
        with open(ref_audio_orig, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_hash = hashlib.md5(audio_data).hexdigest()

        global _ref_audio_cache

        if audio_hash in _ref_audio_cache:
            show_info("Using cached preprocessed reference audio...")
            ref_audio = _ref_audio_cache[audio_hash]

        else:  # first pass, do preprocess
            with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
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
            progress=tqdm,
            streaming=False,
            chunk_size=2048,
    ):
        vocoder = self.vocoder
        device = self.device
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)

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
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                del _

                generated = generated.to(torch.float32)
                generated = generated[:, ref_audio_len:, :]
                generated = generated.permute(0, 2, 1)
                generated_wave = vocoder.decode(generated)
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
            for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
                for chunk in process_batch(gen_text):
                    yield chunk
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
                for future in progress.tqdm(futures) if progress is not None else futures:
                    result = future.result()
                    if result:
                        generated_wave, generated_mel_spec = next(result)
                        generated_waves.append(generated_wave)
                        spectrograms.append(generated_mel_spec)

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

    def infer(self,
              ref_audio,
              ref_text,
              gen_text,
              ):

        audio, sr = torchaudio.load(ref_audio)
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed)
        gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
        for i, gen_text in enumerate(gen_text_batches):
            print(f"gen_text {i}", gen_text)
        print("\n")

        audio_segment, final_sample_rate, spectrogram = next(
            self.infer_batch_process(
                (audio, sr),
                ref_text,
                gen_text_batches
            )
        )
        print(audio_segment)
        print(spectrogram)
        print('Done!')
        return audio_segment, final_sample_rate, spectrogram



if __name__ == "__main__":
    inference = Inference()
    print('Loaded')

    ref_text = 'A meaningful livelihood.'
    ref_audio = 'reference.wav'
    gen_text = 'Today is a good day.'
    audio_segment, final_sample_rate, spectrogram  = inference.infer(ref_audio=ref_audio, ref_text=ref_text, gen_text = gen_text)
    sf.write('output.wav', audio_segment, final_sample_rate)




