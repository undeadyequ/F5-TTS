"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import math
from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


def sde_step_with_logprob(
    x: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    noise_level: float = 0.7,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    sde_type: str = "sde",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Euler-Maruyama SDE step for flow matching with log-probability.

    F5-TTS convention: t ∈ [0,1], x_t = (1-t)·x_noise + t·x_data, v = x_data - x_noise.

    sde_type='sde'  (arxiv 2505.05470):
        σ = noise_level·√(t/(1-t))
        x_{t+dt} = x_t + [v + σ²/(2t)·(x_t + (1-t)·v)]·dt + σ·√dt·ε

    sde_type='cps'  (Coupled Path Sampling):
        σ = (1-t_next)·sin(noise_level·π/2)
        x_next_mean = pred_x_data·t_next + pred_x_noise·√((1-t_next)²-σ²)
        x_next = x_next_mean + σ·ε

    Args:
        x           : [B, N, D]  current state
        v           : [B, N, D]  velocity prediction
        t           : scalar tensor, current timestep
        t_next      : scalar tensor, next timestep (t_next > t)
        noise_level : stochasticity scale (hyperparameter)
        prev_sample : if provided, skip sampling and compute log_prob for this sample
        generator   : optional RNG for reproducibility
        sde_type    : 'sde' or 'cps'

    Returns:
        x_next      : [B, N, D]  next state
        log_prob    : [B]         log-prob of x_next under the step distribution
        x_next_mean : [B, N, D]  deterministic mean
        std_dev_t   : scalar tensor, noise std used in this step
    """
    x = x.float()
    v = v.float()
    t_val = t.float().item()
    t_next_val = t_next.float().item()
    dt = t_next_val - t_val  # positive (F5-TTS: 0→1)
    eps = 1e-6

    if sde_type == "sde":
        # σ_t = noise_level·√(t/(1-t)); 0 at t=0 → pure ODE start
        std_dev_t = noise_level * math.sqrt(t_val / max(1.0 - t_val, eps)) if t_val > eps else 0.0
        std_dev_t_tensor = x.new_tensor(std_dev_t)

        # Drift: v + σ²/(2t)·(x + (1-t)·v)
        if t_val > eps:
            extra = (std_dev_t ** 2 / (2.0 * t_val)) * (x + (1.0 - t_val) * v)
        else:
            extra = torch.zeros_like(x)
        x_next_mean = x + (v + extra) * dt

        # Noise: σ·√dt·ε
        sqrt_dt = math.sqrt(dt)
        if prev_sample is None:
            noise = (
                torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
                if generator is not None
                else torch.randn_like(x)
            )
            x_next = x_next_mean + std_dev_t_tensor * sqrt_dt * noise
        else:
            x_next = prev_sample.float()

        # log N(x_next; x_next_mean, (σ·√dt)²)
        std_sqrt_dt = std_dev_t_tensor * sqrt_dt + eps
        log_prob = (
            -((x_next.detach() - x_next_mean) ** 2) / (2.0 * std_sqrt_dt ** 2)
            - torch.log(std_sqrt_dt)
            - math.log(math.sqrt(2.0 * math.pi))
        )

    elif sde_type == "cps":
        # Predict endpoints from current (x, v)
        pred_x_data  = x + (1.0 - t_val) * v   # predicted x_1 (clean data)
        pred_x_noise = x - t_val * v             # predicted x_0 (noise)

        # σ = (1-t_next)·sin(noise_level·π/2)
        std_dev_t = (1.0 - t_next_val) * math.sin(noise_level * math.pi / 2.0)
        std_dev_t_tensor = x.new_tensor(std_dev_t)

        # CPS mean: blend predicted endpoints toward t_next target
        coeff = math.sqrt(max((1.0 - t_next_val) ** 2 - std_dev_t ** 2, 0.0))
        x_next_mean = pred_x_data * t_next_val + pred_x_noise * coeff

        if prev_sample is None:
            noise = (
                torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
                if generator is not None
                else torch.randn_like(x)
            )
            x_next = x_next_mean + std_dev_t_tensor * noise
        else:
            x_next = prev_sample.float()

        # Unnormalised log-prob (constants dropped, matching SD3 CPS convention)
        log_prob = -((x_next.detach() - x_next_mean) ** 2)

    else:
        raise ValueError(f"Unknown sde_type: {sde_type!r}. Expected 'sde' or 'cps'.")

    # Mean over all dims except batch → [B]
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return x_next, log_prob, x_next_mean, std_dev_t_tensor


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    @torch.no_grad()
    def sde_sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        noise_level: float = 0.7,
        sde_type: str = "sde",
    ):
        """SDE sampling with per-step log-probability tracking.

        Replaces the ODE integrator in sample() with an Euler-Maruyama SDE loop
        using sde_step_with_logprob. Useful for RLHF/GRPO training.

        Returns:
            out           : [B, N, D] synthesised mel (or waveform if vocoder provided)
            all_latents   : list of [B, N, D], length steps+1 (initial noise + each step)
            all_log_probs : list of [B], length steps
        """
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if batch > 1:
            mask = lens_to_mask(duration)
        else:
            mask = None

        # noise input
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:
            timesteps = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            timesteps = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            timesteps = timesteps + sway_sampling_coef * (torch.cos(torch.pi / 2 * timesteps) - 1 + timesteps)

        # velocity prediction (mirrors fn() in sample())
        def predict_v(t_scalar, x):
            x = x.to(step_cond.dtype)  # sde_step_with_logprob returns float32; restore model dtype
            if cfg_strength < 1e-5:
                return self.transformer(
                    x=x, cond=step_cond, text=text, time=t_scalar,
                    mask=mask, drop_audio_cond=False, drop_text=False, cache=True,
                )
            pred_cfg = self.transformer(
                x=x, cond=step_cond, text=text, time=t_scalar,
                mask=mask, cfg_infer=True, cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # SDE loop
        latents = y0
        all_latents = [latents]
        all_log_probs = []

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            v = predict_v(t_curr, latents)

            latents, log_prob, _, _ = sde_step_with_logprob(
                latents, v, t_curr, t_next,
                noise_level=noise_level,
                sde_type=sde_type,
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

        self.transformer.clear_cache()

        out = latents.to(dtype=step_cond.dtype)
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, all_latents, all_log_probs

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
