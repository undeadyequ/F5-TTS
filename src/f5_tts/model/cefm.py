"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension

CEFM: Conditional Energy Flow Matching
Extends CFM with energy-guided loss weighting based on pitch deviation.

New components vs CFM:
  0. uv_mask        — voiced/unvoiced mask per phoneme, from dataloader [B, Tp]
  1. text_aligner   — ASRCNN-style forced-alignment, returns s2s_attn [B, Tp, Tf]
  2. pitch_extractor— JDCNet-style frame-level F0 extractor [B, Tf]
  3. calculate_phoneme_pitch_mean — maps frame F0 → phoneme F0 [B, Tp]
  4. calculate_energy_softmax     — pitch-deviation energy weighting [B]
  5. energy-guided loss           — (energy_softmax * loss_per_sample).sum()
"""
# ruff: noqa: F722 F821

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Helper functions (adapted from /home/rosen/Project/energyDiT/utils.py)
# ---------------------------------------------------------------------------

def calculate_phoneme_pitch_mean(
    f0_gd_frame: torch.Tensor,   # [B, Tf]  frame-level F0 in Hz
    f2p_attn_gd: torch.Tensor,   # [B, Tp, Tf_down]  phoneme-to-frame attention
    eps: float = 1e-8,
) -> torch.Tensor:                # [B, Tp]  phoneme-level mean F0
    """Average frame-level F0 into phoneme-level using f2p attention."""
    # Upsample attention time axis to match f0_gd_frame length
    attn_f2p = F.interpolate(f2p_attn_gd, size=f0_gd_frame.size(-1), mode="nearest")
    w = attn_f2p.transpose(1, 2).to(f0_gd_frame.dtype)  # [B, Tf, Tp]
    denom = w.sum(dim=1).clamp_min(1.0)                  # [B, Tp]
    f0_gd_phn = torch.einsum("bt,btp->bp", f0_gd_frame, w) / (denom + eps)  # [B, Tp]
    return f0_gd_phn

def calculate_phoneme_pitch_curv(
    f0_gd_frame: torch.Tensor,   # [B, Tf]  frame-level F0 in Hz
    f2p_attn_gd: torch.Tensor,   # [B, Tp, Tf_down]  phoneme-to-frame attention
    eps: float = 1e-8,
) -> torch.Tensor:                # [B, Tp]  phoneme-level mean |d²F0/dt²| in semitones
    """Phoneme-level pitch curvature.

    Mirrors calculate_spk_independent_ood_target_v2.py:
        f0_semitones = 12 * log2(f0 / 100)
        d2 = np.gradient(np.gradient(f0_smoothed[start:end]))
        kappa = np.nanmean(np.abs(d2))
    1. Interpolate unvoiced (f0 <= 50) frames of frame-level f0.
    2. Semitone conversion (12 * log2(f0 / 100)) of f0.
    3. Savitzky-Golay smoothing of f0.
    4. Second derivative of smoothed pitch (|d²f0/dt²|)
    5. phoneme level pitch curvature via the frame2phoneme attention

    """
    # Step 1: Interpolate over unvoiced gaps (f0=0) in Hz — mirrors:
    #   valid_idx = np.where(~np.isnan(f0_semitones))[0]
    #   interp_func = scipy.interpolate.interp1d(valid_idx, f0[valid_idx], kind='cubic', ...)
    #   f0_interp = interp_func(np.arange(len(f0)))

    # 1. Linear interpolation (no cubic in PyTorch) as a close approximation.
    f0_interp = f0_gd_frame.clone()  # [B, Tf]
    Tf = f0_gd_frame.size(-1)
    t_idx = torch.arange(Tf, dtype=f0_gd_frame.dtype, device=f0_gd_frame.device)
    for b in range(f0_gd_frame.size(0)):
        voiced = f0_interp[b] > 0                              # [Tf] bool
        if voiced.any() and (~voiced).any():
            v_idx = t_idx[voiced]                              # voiced frame positions
            v_val = f0_interp[b][voiced]                       # voiced F0 values (Hz)
            # torch.searchsorted to find bracketing voiced frames for each unvoiced frame
            u_idx = t_idx[~voiced]
            pos = torch.searchsorted(v_idx, u_idx)             # insertion points
            lo = (pos - 1).clamp(0, len(v_idx) - 1)
            hi = pos.clamp(0, len(v_idx) - 1)
            lo_t, hi_t = v_idx[lo], v_idx[hi]
            lo_v, hi_v = v_val[lo], v_val[hi]
            # linear interpolation; at boundaries lo==hi so value is copied directly
            span = (hi_t - lo_t).clamp(min=1).to(f0_gd_frame.dtype)
            w = ((u_idx - lo_t) / span).clamp(0.0, 1.0)
            f0_interp[b][~voiced] = lo_v + w * (hi_v - lo_v)

    # Step 2: Savitzky-Golay smoothing on Hz values: window=7, polyorder=2.
    # SG(window=7, poly=2) smoothing coefficients: [-2, 3, 6, 7, 6, 3, -2] / 21
    sg_coeffs = torch.tensor([-2., 3., 6., 7., 6., 3., -2.],
                              dtype=f0_interp.dtype, device=f0_interp.device) / 21.0
    f0_smooth = F.conv1d(
        F.pad(f0_interp.unsqueeze(1), (3, 3), mode="replicate"),  # [B, 1, Tf+6]
        sg_coeffs.view(1, 1, -1),                                  # [1, 1, 7]
    ).squeeze(1)                                                   # [B, Tf]

    # Step 3: Second derivative of smoothed Hz F0: d2[t] = f[t-1] - 2f[t] + f[t+1]
    f0_padded = F.pad(f0_smooth, (1, 1), mode="replicate")        # [B, Tf+2]
    d2 = f0_padded[:, :-2] - 2.0 * f0_padded[:, 1:-1] + f0_padded[:, 2:]  # [B, Tf]
    d2_abs = d2.abs()                                                   # [B, Tf]

    # Aggregate frame-level |d²| → phoneme-level mean using f2p attention
    attn_f2p = F.interpolate(f2p_attn_gd, size=f0_gd_frame.size(-1), mode="nearest")
    w = attn_f2p.transpose(1, 2).to(f0_gd_frame.dtype)                 # [B, Tf, Tp]
    denom = w.sum(dim=1).clamp_min(1.0)                                 # [B, Tp]
    curv_phn = torch.einsum("bt,btp->bp", d2_abs, w) / (denom + eps)   # [B, Tp]
    return curv_phn


def calculate_energy_softmax(
    p_metric: torch.Tensor,                    # [B, Tp]  phoneme-level metric (Hz)
    ood_target: float | torch.Tensor,          # scalar or [B]  per-cluster or global target
    uv_mask: torch.Tensor,                     # [B, Tp]  1=voiced, 0=unvoiced
    beta: float = 10.0,
    energy_type: str = "q95_phonemes"           # all_phonemes, q95_phonemes
) -> torch.Tensor:                             # [B]  energy weights summing to B
    """
    Energy-softmax sample weighting based on metric deviation from ood_target.
    ood_target can be a scalar (global) or a [B] tensor (per-sample/cluster).

    For each sample:
    1. compute mean semitone f0 of voiced phonemes given uv_mask
    2. compute energy given mean semitone f0 and target semitone f0
    3. z-score normalise across batch,
    4. apply softmax with temperature beta scaled by batch size so weights sum to B.
    """
    B, Tp = p_metric.shape
    p_metric_voiced = p_metric * uv_mask
    energy_list = []
    for b in range(p_metric_voiced.shape[0]):
        voiced_vals = p_metric_voiced[b][p_metric_voiced[b] != 0]
        Tp_voiced = voiced_vals.shape[0]
        voiced_semi = 12.0 * torch.log2(voiced_vals.clamp(min=1e-6) / 100.0)
        ood_semi = ood_target[b].item() if isinstance(ood_target, torch.Tensor) else float(ood_target)
        #ood_semi = 12.0 * torch.log2(torch.tensor(target, dtype=voiced_vals.dtype, device=voiced_vals.device).clamp(min=1e-6) / 100.0)
        # compute energy on the phonemes whose phoneme-level pitch mean > Q95, instead of all phonemes
        if energy_type == "all_phonemes":
            energy = torch.clamp(ood_semi - voiced_semi, min=0.0).sum() / max(Tp_voiced, 1)
        else:
            q95_threshold = torch.quantile(voiced_semi, 0.95)
            # 筛选 K_top (>= Q95 的音素)
            k_top_vals = voiced_semi[voiced_semi >= q95_threshold]
            k_top_mean = k_top_vals.mean()
            # 获取该 sample 对应的 kappa* (ood_target)
            kappa_star = ood_target[b].item() if isinstance(ood_target, torch.Tensor) else float(ood_target)

            #energy = torch.clamp(kappa_star - k_top_mean, min=0.0)
            energy = kappa_star - k_top_mean

        energy_list.append(energy)
    energy_batch = torch.stack(energy_list)  # [B]
    #print("energy:", energy_batch)
    mean = energy_batch.mean()
    std = energy_batch.std(unbiased=False).clamp(min=1e-6)
    z_energy = (energy_batch - mean) / std
    #print("z-score normalized energy:", z_energy)

    weights = torch.softmax(-beta * z_energy, dim=-1)
    actual_ess = 1.0 / (weights ** 2).sum()
    final_weights = B * weights
    return final_weights, actual_ess  # [B] sums to B, scalar


def calculate_energy_softmax_mu_kappa(
    p_f0_mean: torch.Tensor,                   # [B, Tp]  phoneme-level mean F0 (Hz)
    mu_star: float | torch.Tensor,             # scalar or [B]
    p_f0_curv: torch.Tensor,                   # [B, Tp]  phoneme-level pitch curvature
    kappa_star: float | torch.Tensor,          # scalar or [B]
    uv_mask: torch.Tensor,                     # [B, Tp]  1=voiced, 0=unvoiced
    beta: float = 10.0,
) -> torch.Tensor:                             # [B]  energy weights summing to B
    """Combined mu+kappa energy softmax.
    Adds raw mu-energy and kappa-energy scores per sample before z-score/softmax.
    """
    def _raw_energy(p_metric, ood_target):
        p_voiced = p_metric * uv_mask
        scores = []
        for b in range(p_voiced.shape[0]):
            vals = p_voiced[b][p_voiced[b] != 0]
            Tp_v = vals.shape[0]
            semi = 12.0 * torch.log2(vals.clamp(min=1e-6) / 100.0)
            target = ood_target[b].item() if isinstance(ood_target, torch.Tensor) else float(ood_target)
            ood_s = 12.0 * torch.log2(
                torch.tensor(target, dtype=vals.dtype, device=vals.device).clamp(min=1e-6) / 100.0
            )
            scores.append(torch.clamp(ood_s - semi, min=0.0).sum() / max(Tp_v, 1))
        return torch.stack(scores)  # [B]

    energy_batch = _raw_energy(p_f0_mean, mu_star) + _raw_energy(p_f0_curv, kappa_star)
    mean = energy_batch.mean()
    std = energy_batch.std(unbiased=False).clamp(min=1e-6)
    energy_batch = (energy_batch - mean) / std
    weights = torch.softmax(-beta * energy_batch, dim=-1)
    actual_ess = 1.0 / (weights ** 2).sum()
    gamma = len(energy_batch)
    return gamma * weights, actual_ess  # [B] sums to B, scalar


# ---------------------------------------------------------------------------
# CEFM model
# ---------------------------------------------------------------------------
class CEFM(nn.Module):
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
        # --- energy guidance ---
        text_aligner: nn.Module | None = None,    # ASRCNN-style phoneme-frame aligner
        pitch_extractor: nn.Module | None = None, # JDCNet-style F0 extractor
        mu_star: float = 350.0,                   # OOD pitch target in Hz (global fallback)
        kappa_star: float = 1.60,                 # OOD curvature target (global fallback)
        beta: float = 1.0,                        # energy softmax temperature
        energy_function_type: str = "mu",         # "mu" | "kappa" | "mu_kappa"
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

        # energy guidance modules
        self.text_aligner = text_aligner
        self.pitch_extractor = pitch_extractor
        self.mu_star = mu_star
        self.kappa_star = kappa_star
        self.beta = beta
        self.energy_function_type = energy_function_type

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

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        uv_mask: torch.Tensor | None = None,          # [B, Tp] voiced/unvoiced mask from dataloader
        mel_80: torch.Tensor | None = None,           # [B, 80, T'] ground-truth mel for aux models
        phoneme_tokens: torch.Tensor | None = None,   # [B, Tp] IPA→ASRCNN vocab indices (LongTensor)
        mu_stars: torch.Tensor | None = None,         # [B] per-sample mu_star (overrides self.mu_star)
        kappa_stars: torch.Tensor | None = None,      # [B] per-sample kappa_star (overrides self.kappa_star)
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
        loss = F.mse_loss(pred, flow, reduction="none")  # [B, N, D]

        # -------------------------------------------------------------------
        # Energy-guided loss (training only, requires all three components)
        # -------------------------------------------------------------------
        if (
            self.text_aligner is not None
            and self.pitch_extractor is not None
            and uv_mask is not None
            and mel_80 is not None
            and phoneme_tokens is not None
            and self.training
        ):
            # mel_80: [B, 80, T'] — ground-truth mel computed with energyDiT preprocess()
            # (n_fft=2048, win_length=1200, hop_length=300, log-norm mean=-4, std=4)
            # Using this instead of inp avoids distribution mismatch with aux model training.
            mels_80 = mel_80.to(inp.dtype)

            # Step 1: Force alignment
            # Mirrors StyleTTS2/train_first_txt2mel_cfm_icl.py:
            #   n_down = model.text_aligner.n_down  # 1
            #   mask = length_to_mask(mel_input_length // (2 ** n_down))
            #   ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)
            # mask spans ASRCNN encoder output time (mel_80 length // 2), True = padding.
            # mel_80 uses hop=300; F5-TTS mel uses hop=256 on the same audio:
            #   mel_80_lens = ceil(lens * 256 / 300), clamped to T_80
            n_down = self.text_aligner.n_down  # 1 for ASRCNN
            T_80 = mels_80.shape[-1]
            mel_80_lens = (lens.float() * 256 / 300).ceil().long().clamp(max=T_80)
            # Actual encoder output size from init_cnn (kernel=7, padding=3, stride=2):
            #   T_enc = floor((T_80 - 1) / 2) + 1
            # Must use T_enc as mask length — enc_lens.max() can be off-by-one due to
            # hop-rate approximation, causing masked_fill_ shape mismatch.
            T_enc = (T_80 - 1) // 2 + 1
            enc_lens = (mel_80_lens // (2 ** n_down)).clamp(max=T_enc)
            # StyleTTS2 length_to_mask: gt(index + 1, length) → True = padding
            asr_idx = torch.arange(T_enc, device=device).unsqueeze(0).expand(batch, -1)
            asr_pad_mask = torch.gt(asr_idx + 1, enc_lens.unsqueeze(1))
            ppgs, s2s_pred, s2s_attn = self.text_aligner(mels_80, asr_pad_mask, phoneme_tokens)

            # Post-process: remove BOS token phoneme dimension
            s2s_attn = s2s_attn.transpose(-1, -2)   # [B, Tf, Tp]
            s2s_attn = s2s_attn[..., 1:]            # [B, Tf, Tp-1]  drop BOS
            s2s_attn = s2s_attn.transpose(-1, -2)   # [B, Tp-1, Tf]

            # Trim uv_mask to match Tp-1 (no cutting needed in F5-TTS since mel is not cut)
            uv_mask_trim = uv_mask[:, :s2s_attn.shape[1]]  # [B, Tp-1]

            # Step 2: Frame-level pitch extraction
            # JDCNet forward(x [B, 1, mel, T]) → (F0 [B, Tf], GAN_feat, poolblock)
            F0_real, _, _ = self.pitch_extractor(mels_80.unsqueeze(1))  # F0_real: [B, Tf]

            # Step 3: Phoneme-level features
            # Resolve per-sample targets: batch tensor takes priority over scalar fallback
            _mu = mu_stars if mu_stars is not None else self.mu_star
            _kappa = kappa_stars if kappa_stars is not None else self.kappa_star

            # Step 4: Energy softmax weights [B] — dispatch by energy_function_type
            if self.energy_function_type == "mu":
                p_f0_mean = calculate_phoneme_pitch_mean(F0_real, s2s_attn)  # [B, Tp-1]
                energy_softmax, actual_ess = calculate_energy_softmax(
                    p_f0_mean, _mu, uv_mask_trim, beta=self.beta, energy_type="q95_phonemes")
            elif self.energy_function_type == "kappa":
                p_f0_curv = calculate_phoneme_pitch_curv(F0_real, s2s_attn)  # [B, Tp-1]
                energy_softmax, actual_ess = calculate_energy_softmax(
                    p_f0_curv, _kappa, uv_mask_trim, beta=self.beta
                )
            elif self.energy_function_type == "mu_kappa":
                p_f0_mean = calculate_phoneme_pitch_mean(F0_real, s2s_attn)  # [B, Tp-1]
                p_f0_curv = calculate_phoneme_pitch_curv(F0_real, s2s_attn)  # [B, Tp-1]
                energy_softmax, actual_ess = calculate_energy_softmax_mu_kappa(
                    p_f0_mean, _mu, p_f0_curv, _kappa, uv_mask_trim, beta=self.beta
                )
            else:
                raise ValueError(f"Unknown energy_function_type: {self.energy_function_type!r}")

            # Step 5: Per-sample loss over masked region, then energy-weighted sum
            loss_masked = loss * rand_span_mask.unsqueeze(-1)                       # [B, N, D]
            n_per_sample = rand_span_mask.sum(dim=-1).clamp(min=1).float()          # [B]
            loss_per_sample = loss_masked.sum(dim=[1, 2]) / (n_per_sample * self.num_channels)  # [B]
            loss_cefm = (energy_softmax * loss_per_sample).sum()                    # scalar

            return loss_cefm, cond, pred, actual_ess

        # -------------------------------------------------------------------
        # Fallback: standard CFM loss (no aligner / extractor / uv_mask)
        # -------------------------------------------------------------------
        loss = loss[rand_span_mask]
        return loss.mean(), cond, pred, None
