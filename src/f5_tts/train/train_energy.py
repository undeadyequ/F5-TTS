# training script for CEFM (energy-guided flow matching).
# Modification of train.py — uses CEFM instead of CFM and adds:
#   - EnergyDataset     : wraps CustomDataset with mel_80 / uv_mask / phoneme_tokens
#   - collate_fn_energy : pads energy fields alongside mel/text in each batch
#   - EnergyTrainer     : passes mel_80 / uv_mask / phoneme_tokens into model.forward()
#
# External model requirements
# ---------------------------
# text_aligner  : ASRCNN from energyDiT  (Utils/ASR/models.py)
# pitch_extractor: JDCNet from energyDiT (Utils/JDC/model.py)
# Both are loaded from checkpoints and frozen (eval, no grad) during training.
# Set the following in the hydra config under the `energy:` section:
#   energy.F0_path  : path to JDCNet checkpoint (bst.t7), relative to src/
#   energy.ASR_path : path to ASRCNN checkpoint (.pth), relative to src/
#   energy.ASR_config: path to ASRCNN config.yml, relative to src/
#   energy.mu_star  : OOD pitch target in Hz (default 350.0)
#   energy.beta     : energy softmax temperature (default 10.0)

import math
import os
from importlib.resources import files

import hydra
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from f5_tts.model import Trainer
from f5_tts.model.cefm import CEFM
from f5_tts.model.dataset import DynamicBatchSampler, load_dataset
from f5_tts.model.dataset_energy import EnergyDataset, collate_fn_energy
from f5_tts.model.utils import exists, get_tokenizer
import logging

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project
logging.getLogger("phonemizer").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# EnergyTrainer
# ---------------------------------------------------------------------------

class EnergyTrainer(Trainer):
    """Trainer subclass for CEFM energy-guided training.

    Changes vs base Trainer:
      1. load_checkpoint: strips text_aligner / pitch_extractor keys before loading
         so that an original F5-TTS checkpoint (which has no auxiliary models) can be
         resumed without a key-mismatch error. Mirrors StyleTTS2's ignore_modules pattern.
      2. DataLoader uses collate_fn_energy (adds mel_80 / uv_mask / phoneme_tokens).
      3. model.forward() receives mel_80 / uv_mask / phoneme_tokens from batch.
    """

    # Submodule prefixes to ignore when loading checkpoints from base F5-TTS.
    # Keys starting with any of these are stripped before load_state_dict().
    _IGNORE_MODULES = ("text_aligner.", "pitch_extractor.")

    def load_checkpoint(self):
        """Override to strip auxiliary-model keys before loading state_dicts.

        This allows resuming from an original F5-TTS checkpoint that has no
        text_aligner / pitch_extractor entries, using strict=False so that
        missing keys (the aux models) are silently ignored while unexpected
        keys cause no error either.
        """
        import gc  # noqa: PLC0415

        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415
            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        else:
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        def _filter(state_dict):
            """Remove aux-model keys and backward-compat mel_spec keys."""
            return {
                k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in self._IGNORE_MODULES)
                and k not in ("ema_model.mel_spec.mel_stft.mel_scale.fb",
                              "ema_model.mel_spec.mel_stft.spectrogram.window")
            }

        if self.is_main:
            ema_sd = _filter(checkpoint["ema_model_state_dict"])
            missing, unexpected = self.ema_model.load_state_dict(ema_sd, strict=False)
            aux_missing = [k for k in missing if any(k.startswith(p) for p in ("ema_model.text_aligner.", "ema_model.pitch_extractor."))]
            other_missing = [k for k in missing if k not in aux_missing]
            if other_missing:
                print(f"EnergyTrainer: unexpected missing keys in EMA: {other_missing}")
            if aux_missing:
                print(f"EnergyTrainer: {len(aux_missing)} aux-model keys not in checkpoint (expected when resuming from base F5-TTS).")

        if "update" in checkpoint or "step" in checkpoint:
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
            model_sd = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if not any(k.startswith(p) for p in self._IGNORE_MODULES)
                and k not in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window")
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(model_sd, strict=False)
            update = checkpoint["update"]  # always restore update count
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError:
                # Optimizer param groups don't match — checkpoint is from a different
                # model architecture (e.g. base F5-TTS without aux models).
                # Keep the update count but start optimizer state fresh.
                print("EnergyTrainer: optimizer state incompatible with current model (different param groups). Starting optimizer fresh.")
        else:
            model_sd = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ("initted", "update", "step")
                and not any(k.replace("ema_model.", "").startswith(p) for p in self._IGNORE_MODULES)
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(model_sd, strict=False)
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset, num_workers=16, resumable_with_seed=None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_energy,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn_energy,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(
                f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
            )

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(train_dataloader, self.scheduler)
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(
                train_dataloader.batch_sampler, "set_epoch"
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    mel_80 = batch.get("mel_80")                  # [B, 80, T'] or None
                    uv_mask = batch.get("uv_mask")                # [B, Tp] or None
                    phoneme_tokens = batch.get("phoneme_tokens")  # [B, Tp] LongTensor or None
                    mu_stars = batch.get("mu_stars")              # [B] or None
                    kappa_stars = batch.get("kappa_stars")        # [B] or None

                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred, actual_ess = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
                        noise_scheduler=self.noise_scheduler,
                        mel_80=mel_80,
                        uv_mask=uv_mask,
                        phoneme_tokens=phoneme_tokens,
                        mu_stars=mu_stars,
                        kappa_stars=kappa_stars,
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    log_dict = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                    if actual_ess is not None:
                        log_dict["ess"] = actual_ess.item()
                    self.accelerator.log(log_dict, step=global_update)
                if self.logger == "tensorboard" and self.accelerator.is_main_process:
                    self.writer.add_scalar("loss", loss.item(), global_update)
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode(), self.accelerator.autocast():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = (
                                generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            )
                            ref_mel_spec = batch["mel"][0, :, :ref_audio_len].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )
                        self.model.train()

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()


# ---------------------------------------------------------------------------
# Helper: load frozen auxiliary models from energyDiT
# ---------------------------------------------------------------------------

def _load_text_aligner(ckpt_path: str, asr_config_path: str) -> torch.nn.Module:
    """Load pretrained ASRCNN text aligner (mirrors load_ASR_models in energyDiT)."""
    import yaml  # noqa: PLC0415
    from third_party.Utils.ASR.models import ASRCNN  # noqa: PLC0415

    with open(asr_config_path, "r") as f:
        model_config = yaml.safe_load(f)["model_params"]

    model = ASRCNN(**model_config)
    params = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(params["model"])
    model.eval().requires_grad_(False)
    return model


def _load_pitch_extractor(ckpt_path: str) -> torch.nn.Module:
    """Load pretrained JDCNet pitch extractor.

    bst.t7 is a Torch7/Lua serialized file — requires weights_only=False.
    num_class=1 selects the regression variant (single F0 value per frame).
    """
    from third_party.Utils.JDC.model import JDCNet  # noqa: PLC0415

    model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(params["net"])
    model.eval().requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    wandb_project = model_cfg.ckpts.get("wandb_project", "CEFM-TTS")
    wandb_run_name = model_cfg.ckpts.get(
        "wandb_run_name",
        f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}",
    )
    wandb_resume_id = model_cfg.ckpts.get("wandb_resume_id", None)

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # energy config — key names match F5TTS_v1_Small_energy.yaml
    # Paths in the yaml are relative to src/ (where third_party/ lives).
    _src = str(files("f5_tts").joinpath(".."))  # .../src/f5_tts/.. = .../src
    def _resolve(p):
        return os.path.join(_src, p) if p is not None else None

    energy_cfg = model_cfg.get("energy", {})
    asr_config_path = _resolve(energy_cfg.get("ASR_config", None))
    text_aligner_ckpt = _resolve(energy_cfg.get("ASR_path", None))
    pitch_extractor_ckpt = _resolve(energy_cfg.get("F0_path", None))
    mu_star = float(energy_cfg.get("mu_star", 21.69))  # 12.0 * torch.log2(torch.tensor(350.0) / 100.0)
    kappa_star = float(energy_cfg.get("kappa_star", 1.60))
    beta = float(energy_cfg.get("beta", 10.0))
    energy_function_type = str(energy_cfg.get("energy_function_type", "mu"))
    spk_clusters_path = energy_cfg.get("spk_clusters_path", None)

    # load auxiliary frozen models
    text_aligner = None
    pitch_extractor = None

    if text_aligner_ckpt is not None and asr_config_path is not None:
        print(f"Loading text aligner from {text_aligner_ckpt}")
        text_aligner = _load_text_aligner(text_aligner_ckpt, asr_config_path)

    if pitch_extractor_ckpt is not None:
        print(f"Loading pitch extractor from {pitch_extractor_ckpt}")
        pitch_extractor = _load_pitch_extractor(pitch_extractor_ckpt)

    if text_aligner is None or pitch_extractor is None:
        print(
            "WARNING: text_aligner or pitch_extractor not loaded. "
            "CEFM will fall back to standard CFM loss (energy guidance disabled). "
            "Set energy.F0_path, energy.ASR_path, energy.ASR_config in the config to enable."
        )

    # set model
    model = CEFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        text_aligner=text_aligner,
        pitch_extractor=pitch_extractor,
        mu_star=mu_star,
        kappa_star=kappa_star,
        beta=beta,
        energy_function_type=energy_function_type,
    )

    # init trainer
    trainer = EnergyTrainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = EnergyDataset(
        load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec),
        spk_clusters_path=spk_clusters_path,
    )
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()
