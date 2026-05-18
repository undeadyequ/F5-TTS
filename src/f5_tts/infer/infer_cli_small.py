"""Inference script for local F5TTS Small models.

Simpler than infer_cli.py: no TOML config, no HuggingFace downloads, no multi-voice.
Checkpoints are loaded from /home/rosen/Project/F5-TTS/ckpts/{model_name}/.

Usage:
    python src/f5_tts/infer/infer_cli_small.py \
        --model  F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500 \
        --cfg    src/f5_tts/configs/F5TTS_v1_Small.yaml \
        --step   last \
        --ref_audio  /path/to/ref.wav \
        --ref_text   "Reference transcript." \
        --gen_text   "Text to synthesize." \
        --output     /path/to/out.wav
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CKPTS_ROOT = Path("/home/rosen/Project/F5-TTS/ckpts")
DEFAULT_VOCAB = "/home/rosen/Project/F5-TTS/data/LibriTTS_100_360_500_char/vocab.txt"
CONFIGS_DIR = Path(__file__).parents[1] / "configs"  # src/f5_tts/configs/

# Mapping from model-name prefix → config yaml stem.
# The model name encodes the experiment version (v4, v5, v6…) but the configs
# are all named F5TTS_v1_Small_energy_v{N}.yaml — so we need an explicit map.
MODEL_TO_CONFIG = {
    "F5TTS_v1_Small_energy_spkIndepOOD_kappa": "F5TTS_v1_Small_energy_v5",
    "F5TTS_v1_Small_energy_spkIndepOOD":       "F5TTS_v1_Small_energy",
    "F5TTS_v4_Small_energy_spkIndepOOD_mu":    "F5TTS_v1_Small_energy_v4",
    "F5TTS_v5_beta2_Small_energy_spkIndepOOD": "F5TTS_v1_Small_energy_v5_beta2",
    "F5TTS_v5_beta3_Small_energy_spkIndepOOD": "F5TTS_v1_Small_energy_v5_beta3",
    "F5TTS_v5_Small_energy_spkIndepOOD":       "F5TTS_v1_Small_energy_v5",
    "F5TTS_v6_Small_energy_spkIndepOOD":       "F5TTS_v1_Small_energy_v6",
    "F5TTS_v1_Small":                          "F5TTS_v1_Small",
}


def guess_config(model_name: str) -> Path | None:
    """Return the config yaml path inferred from the model directory name."""
    for prefix, cfg_stem in MODEL_TO_CONFIG.items():
        if model_name.startswith(prefix):
            p = CONFIGS_DIR / f"{cfg_stem}.yaml"
            return p if p.exists() else None
    return None


def resolve_ckpt(model_dir: Path, step: str) -> Path:
    if step == "last":
        p = model_dir / "model_last.pt"
    else:
        p = model_dir / f"model_{step}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Inference for local F5TTS Small models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Model selection
parser.add_argument("--model",  required=True,
                    help="Model directory name under ckpts/ (e.g. F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500)")
parser.add_argument("--cfg",    default=None,
                    help="Path to model config yaml. Auto-detected from model name if omitted.")
parser.add_argument("--step",   default="last",
                    help="Checkpoint step number or 'last'.")
parser.add_argument("--vocab",  default=DEFAULT_VOCAB,
                    help="Path to vocab.txt.")
parser.add_argument("--ckpts_root", default=str(CKPTS_ROOT),
                    help="Root directory that contains model subdirectories.")

# Audio I/O
parser.add_argument("--ref_audio", required=True,  help="Reference audio file.")
parser.add_argument("--ref_text",  default="",      help="Transcript of reference audio (auto-transcribed if empty).")
parser.add_argument("--gen_text",  default=None,    help="Text to synthesize.")
parser.add_argument("--gen_file",  default=None,    help="File containing text to synthesize (overrides --gen_text).")
parser.add_argument("--output",    required=True,   help="Output wav file path.")

# Inference knobs
parser.add_argument("--nfe_step",           type=int,   default=nfe_step)
parser.add_argument("--cfg_strength",       type=float, default=cfg_strength)
parser.add_argument("--sway_sampling_coef", type=float, default=sway_sampling_coef)
parser.add_argument("--speed",              type=float, default=speed)
parser.add_argument("--fix_duration",       type=float, default=fix_duration)
parser.add_argument("--cross_fade_duration",type=float, default=cross_fade_duration)
parser.add_argument("--target_rms",         type=float, default=target_rms)
parser.add_argument("--remove_silence",     action="store_true")
parser.add_argument("--device",             default=device)

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
model_dir = Path(args.ckpts_root) / args.model
if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

ckpt_path = resolve_ckpt(model_dir, args.step)

if args.cfg:
    cfg_path = Path(args.cfg)
else:
    cfg_path = guess_config(args.model)
    if cfg_path is None:
        raise ValueError(
            f"Could not auto-detect config for '{args.model}'.\n"
            f"Pass --cfg explicitly. Available configs:\n"
            + "\n".join(f"  {p.name}" for p in sorted(CONFIGS_DIR.glob("F5TTS_v*_Small*.yaml")))
        )

print(f"Model   : {model_dir}")
print(f"Ckpt    : {ckpt_path}")
print(f"Config  : {cfg_path}")
print(f"Vocab   : {args.vocab}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model_cfg = OmegaConf.load(cfg_path)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
vocoder_name = model_cfg.model.mel_spec.mel_spec_type  # "vocos" or "bigvgan"

vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, device=args.device)

ema_model = load_model(
    model_cls, model_arc, str(ckpt_path),
    mel_spec_type=vocoder_name,
    vocab_file=args.vocab,
    device=args.device,
)

# ---------------------------------------------------------------------------
# Text input
# ---------------------------------------------------------------------------
if args.gen_file:
    gen_text = open(args.gen_file, encoding="utf-8").read()
elif args.gen_text:
    gen_text = args.gen_text
else:
    raise ValueError("Provide --gen_text or --gen_file.")

# ---------------------------------------------------------------------------
# Preprocess reference
# ---------------------------------------------------------------------------
ref_audio, ref_text = preprocess_ref_audio_text(args.ref_audio, args.ref_text)

# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------
audio_segment, final_sr, _ = infer_process(
    ref_audio,
    ref_text,
    gen_text,
    ema_model,
    vocoder,
    mel_spec_type=vocoder_name,
    target_rms=args.target_rms,
    cross_fade_duration=args.cross_fade_duration,
    nfe_step=args.nfe_step,
    cfg_strength=args.cfg_strength,
    sway_sampling_coef=args.sway_sampling_coef,
    speed=args.speed,
    fix_duration=args.fix_duration,
    device=args.device,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out_path), audio_segment, final_sr)

if args.remove_silence:
    remove_silence_for_generated_wav(str(out_path))

print(f"Saved → {out_path}")
