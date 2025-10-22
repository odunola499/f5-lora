import os
from datetime import datetime
from random_word import RandomWords
from typing import Optional
import wandb
import random
import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelCheckpoint
)
from lightning.pytorch.loggers import CSVLogger, CometLogger
from ema_pytorch import EMA
from .optimizer import OPTIMIZER_MAPPING, LR_SCHEDULER_MAPPING
from f5_lora.config import Config
from f5_lora.modules.commons import load_model, load_vocoder
from safetensors.torch import save_file
from f5_lora.modules.lora import LoraManager

now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
word = RandomWords().get_random_word()

run_name = f"{now_str}_{word}"


class TrainModule(pl.LightningModule):
    def __init__(
            self,
            config: Config,
            train_loader: DataLoader,
            valid_loader: Optional[DataLoader] = None,
            lora: bool = False,
            alpha: int = 8,
            rank: int = 4,
    ):
        super().__init__()
        self.vocoder = None
        self.model = None
        self.ema_model = None

        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lora = lora
        self.alpha = alpha
        self.rank = rank

    def configure_model(self):
        config = self.config

        if config.train.resume_run:
            model = load_model(
                device=self.device,
                config=config,
                dtype=self.dtype,
                load_pretrained=True,
                ckpt_path=config.train.pretrained_ckpt,
                use_ema=True
            )
        else:
            model = load_model(
                device=self.device,
                config=config,
                dtype=self.dtype,
                load_pretrained=False,
                ckpt_path=None,
                use_ema=True
            )

        if self.lora:
            lora_manager = LoraManager(model)
            lora_manager.prepare(
                rank=self.rank,
                alpha=self.alpha,
                target_modules=None,
                report=True
            )
            model = lora_manager.model
            print("Initialized LoRA modules.")

        model.transformer.text_embed.requires_grad_(False)
        ema_model = EMA(model, include_online_model=False)
        vocoder = load_vocoder(self.device)
        vocoder.requires_grad_(False)

        self.model = model
        self.ema_model = ema_model
        self.vocoder = vocoder

    def configure_optimizers(self):
        optimizer_cls = OPTIMIZER_MAPPING.get(self.config.train.optimizer, None)
        lr_scheduler_cls = LR_SCHEDULER_MAPPING.get(self.config.train.lr_scheduler, None)

        optimizer = optimizer_cls(
            self.parameters(), lr=self.config.train.learning_rate, weight_decay=1e-2
        )
        scheduler = lr_scheduler_cls(
            optimizer,
            num_warmup_steps=self.config.train.warmup_steps,
            num_training_steps=self.config.train.max_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def save(self):
        directory = self.config.train.ckpt_dir
        save_n_files = self.config.train.keep_last_n_checkpoints
        os.makedirs(directory, exist_ok=True)

        state_dict = self.ema_model.state_dict()
        if self.lora:
            state_dict = {k: v for k, v in state_dict.items() if 'lora' in k.lower() and isinstance(v, torch.Tensor)}
            state_dict['alpha'] = torch.tensor(self.alpha)
            state_dict['rank'] = torch.tensor(self.rank)
            filename = os.path.join(directory, f'adapter_{self.global_step}.safetensors')
        else:
            state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            filename = os.path.join(directory, f'model_{self.global_step}.safetensors')

        save_file(state_dict, filename)
        ckpts = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory) if
             f.startswith("model_") or f.startswith("adapter_")],
            key=os.path.getmtime,
        )
        for ckpt in ckpts[:-save_n_files]:
            os.remove(ckpt)

        print(f"Saved checkpoint: {filename}")

    def train_dataloader(self):
        return self.train_loader

    def training_step(self, batch):
        mel_spec = batch['mel'].permute(0, 2, 1)
        mel_lengths = batch['mel_lengths']
        text_inputs = batch['text']

        loss, cond, pred = self.model(
            mel_spec, text=text_inputs, lens=mel_lengths,
        )
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        should_save = (
                self.global_step > 0
                and self.trainer.is_global_zero
                and self.global_step % self.config.train.save_interval == 0
        )
        if should_save:
            self.save()

        return loss

    def validation_step(self, batch):
        mel_spec = batch['mel'].permute(0, 2, 1)
        mel_lengths = batch['mel_lengths']
        text_inputs = batch['text']

        loss, cond, pred = self.model(
            mel_spec, text=text_inputs, lens=mel_lengths,
        )
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        if self.config.train.log_samples and self.global_rank == 0:
            if random.random() < 0.2:
                self.log_audio(batch)

        return loss

    def on_after_backward(self) -> None:
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("train/grad_norm", total_norm, prog_bar=True, sync_dist=True)

    def log_audio(self, batch):
        mel_spec = batch['mel']
        text_inputs = batch['text']
        mel_lengths = batch['mel_lengths']
        ref_audio_len = mel_lengths[0]
        infer_text = [
            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
        ]
        with torch.inference_mode():
            generated, _ = self.model.sample(
                cond=infer_text,
                duration=ref_audio_len * 2,
                steps=self.config.inference.nfe_step,
                cfg_strength=self.config.inference.cfg_strength,
                sway_sampling_coef=self.config.inference.sway_sampling_coef,
            )
            generated = generated.to(torch.float32)
            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
            ref_mel_spec = batch["mel"][0].unsqueeze(0)

            gen_audio = self.vocoder.decode(gen_mel_spec).cpu().to_numpy()
            ref_audio = self.vocoder.decode(ref_mel_spec).cpu().to_numpy()

        self.logger.log('audio/gen_sample', wandb.Audio(gen_audio, caption=infer_text, sample_rate=24000))
        self.logger.log('audio/ref_sample', wandb.Audio(ref_audio, caption=infer_text, sample_rate=24000))

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer) -> None:
        optimizer.zero_grad()
        # if self.global_rank == 0:
        self.ema_model.update()


def train_model(config: Config, train_module: TrainModule):
    config = config
    train_module = train_module

    callbacks = []
    if config.train.val_interval:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    if config.train.log_to == 'comet':
        logger = CometLogger(
            project_name=config.train.wandb_project,
            experiment_name=config.train.wandb_run_name or run_name

        )
    else:
        logger = CSVLogger(save_dir='train_checkpoints')
    checkpoint = ModelCheckpoint(
        'train_checkpoints',
        monitor='train/loss',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.train.epochs,
        max_steps=config.train.max_steps,
        accumulate_grad_batches=config.train.grad_accumulation_steps or 1,
        gradient_clip_val=config.train.max_grad_norm or 0.0,
        log_every_n_steps=4,
        accelerator="auto",
        devices="auto",
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        enable_progress_bar=True,
        enable_model_summary=True,
        benchmark=True,
        num_sanity_val_steps=0 if config.train.val_interval else None,
    )

    trainer.fit(train_module, train_dataloaders=train_module.train_loader, val_dataloaders=train_module.valid_loader)

