"""
EnergyDataset and collate_fn_energy for CEFM energy-guided training.

Extends CustomDataset with three additional fields per item:
  - "phonemes"       : IPA string from espeak (str)
  - "phoneme_tokens" : IPA → ASRCNN vocab indices (LongTensor [Tp])
  - "uv_mask"        : voiced/unvoiced mask, BOS/EOS-padded (FloatTensor [Tp+2])
  - "mel_80"         : 80-bin log-mel with energyDiT normalization ([80, T'])
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import phonemizer as phonemizer_lib

# Suppress phonemizer's word-count mismatch warnings (triggered by number/contraction
# expansion in espeak, e.g. "100" → "wʌn hʌndɹəd"). The phonemization itself is correct.


# ---------------------------------------------------------------------------
# Phonemizer — same backend and settings as energyDiT/modify_train_list_txt.py
# ---------------------------------------------------------------------------

_global_phonemizer = phonemizer_lib.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True,
    logger=None,
)


def _text_to_phonemes(text: str) -> str:
    """Convert raw English text to IPA phoneme string via espeak."""
    return _global_phonemizer.phonemize([text])[0]


# ---------------------------------------------------------------------------
# TextCleaner — maps IPA characters to integer indices for ASRCNN.
# Mirrors meldataset.py in StyleTTS2 / energyDiT exactly.
# symbols = _pad + _punctuation + _letters + _letters_ipa  → 178 tokens
# ASRCNN was trained with this vocab (n_token=178 in config.yml).
# ---------------------------------------------------------------------------

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»\u201c\u201d '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\u0329\u02b0ᵻ"

_symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
_sym2idx = {s: i for i, s in enumerate(_symbols)}


class TextCleaner:
    """Character-level tokenizer for IPA strings using ASRCNN's 178-symbol vocab."""
    def __call__(self, text: str) -> list[int]:
        indexes = []
        for char in text:
            try:
                indexes.append(_sym2idx[char])
            except KeyError:
                pass  # silently skip unknown characters (same as StyleTTS2)
        return indexes


_text_cleaner = TextCleaner()


# ---------------------------------------------------------------------------
# build_voiced_mask — copied from energyDiT/meldataset2.py
# ---------------------------------------------------------------------------

def _build_voiced_mask(ipa_string: str) -> torch.Tensor:
    voiced_ipa = {
        # vowels
        "i", "y", "ɨ", "ʉ", "ɯ", "u",
        "ɪ", "ʏ", "ʊ",
        "e", "ø", "ɘ", "ɵ", "ɤ", "o",
        "ə", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ",
        "æ", "ɐ", "a", "ɶ", "ɑ", "ɒ",
        # voiced consonants
        "b", "d", "ɡ", "v", "ð", "z", "ʒ",
        "ʝ", "ɣ", "ʁ", "ʕ", "ɦ",
        "m", "n", "ŋ", "ɱ", "ɳ", "ɲ", "ŋ̊",
        "l", "ɫ", "ɭ", "ʎ", "r", "ɹ", "ɻ", "ɾ",
        "w", "j",
    }
    return torch.tensor(
        [1.0 if ch in voiced_ipa else 0.0 for ch in ipa_string], dtype=torch.float32
    )


# ---------------------------------------------------------------------------
# mel_80 extractor — same params and normalization as energyDiT/meldataset2.py
# ---------------------------------------------------------------------------

_mel_80_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
_MEL_80_MEAN = -4.0
_MEL_80_STD = 4.0


def _wav_to_mel80(wav: torch.Tensor) -> torch.Tensor:
    """Reproduce energyDiT preprocess(): log-mel with mean/std normalization.

    Args:
        wav: [T] float waveform at 24 kHz
    Returns:
        mel: [80, T'] normalized mel spectrogram
    """
    mel = _mel_80_transform(wav)                               # [80, T']
    mel = (torch.log(1e-5 + mel) - _MEL_80_MEAN) / _MEL_80_STD
    return mel


# ---------------------------------------------------------------------------
# EnergyDataset
# ---------------------------------------------------------------------------

class EnergyDataset(torch.utils.data.Dataset):
    """Thin wrapper around a CustomDataset that also returns:
      - "phonemes"       : IPA string from espeak
      - "phoneme_tokens" : IPA → ASRCNN vocab indices, LongTensor [Tp]
      - "uv_mask"        : voiced/unvoiced mask BOS/EOS-padded, FloatTensor [Tp+2]
      - "mel_80"         : [80, T'] mel with energyDiT-compatible STFT params
      - "mu_star"        : per-sample OOD pitch target (float), from speaker cluster
      - "kappa_star"     : per-sample OOD curvature target (float), from speaker cluster

    The CustomDataset already caches/loads `mel_spec` (F5-TTS, 100-bin vocos).
    We re-load the audio to compute the 80-bin mel with correct STFT params.

    spk_clusters_path: path to JSON file with format
        {"cluster_name": [[mu_star, kappa_star], [spk1, spk2, ...]]}
    If None, mu_star/kappa_star items are not added (trainer uses CEFM's scalar fallback).
    """

    def __init__(self, base_dataset, spk_clusters_path: str | None = None):
        self.base = base_dataset

        # Build speaker → (mu_star, kappa_star) lookup from cluster JSON
        self._spk_to_mu_kappa: dict[str, tuple[float, float]] = {}
        if spk_clusters_path is not None:
            with open(spk_clusters_path, "r") as f:
                clusters = json.load(f)
            for _cluster_name, (targets, speakers) in clusters.items():
                mu, kappa = float(targets[0]), float(targets[1])
                for spk in speakers:
                    self._spk_to_mu_kappa[str(spk)] = (mu, kappa)

    def get_frame_len(self, index):
        return self.base.get_frame_len(index)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        row = self.base.data[index]  # {"audio_path": ..., "text": ..., "duration": ...}

        # --- phonemes ---------------------------------------------------------
        text = row["text"]
        phonemes = _text_to_phonemes(text)  # IPA string
        item["phonemes"] = phonemes

        # --- uv_mask ----------------------------------------------------------
        # BOS/EOS padded with 0 (mirrors energyDiT meldataset2.py)
        uv_mask = _build_voiced_mask(phonemes)         # [len(phonemes)]
        uv_mask = F.pad(uv_mask, (1, 1), value=0.0)   # [len(phonemes) + 2]
        item["uv_mask"] = uv_mask

        # --- phoneme_tokens ---------------------------------------------------
        # IPA chars → ASRCNN vocab indices (same as StyleTTS2 TextCleaner)
        item["phoneme_tokens"] = torch.LongTensor(_text_cleaner(phonemes))

        # --- mu_star / kappa_star from speaker cluster -----------------------
        audio_path = row["audio_path"]
        if self._spk_to_mu_kappa:
            # Speaker ID = grandparent directory name of audio file
            # e.g. .../train-clean-100/125/121124/125_121124_000116_000000.wav → "125"
            spk_id = Path(audio_path).parent.parent.name
            mu_kappa = self._spk_to_mu_kappa.get(spk_id)
            if mu_kappa is not None:
                item["mu_star"] = mu_kappa[0]
                item["kappa_star"] = mu_kappa[1]

        # --- mel_80 -----------------------------------------------------------
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.squeeze(0)  # [T]
        item["mel_80"] = _wav_to_mel80(wav)            # [80, T']

        return item


# ---------------------------------------------------------------------------
# collate_fn_energy
# ---------------------------------------------------------------------------

def collate_fn_energy(batch):
    """collate_fn extended to pad/stack mel_80, uv_mask, and phoneme_tokens."""
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padded_mel_specs.append(F.pad(spec, (0, max_mel_length - spec.size(-1)), value=0))
    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    phonemes = [item.get("phonemes", "") for item in batch]  # list[str]

    # mel_80: [80, T'] per sample — pad to longest in batch
    mel_80 = None
    if batch[0].get("mel_80") is not None:
        m80s = [item["mel_80"] for item in batch]
        max_t = max(m.shape[-1] for m in m80s)
        mel_80 = torch.stack([F.pad(m, (0, max_t - m.shape[-1]), value=0.0) for m in m80s])

    # uv_mask: [B, Tp] float
    uv_mask = None
    if batch[0].get("uv_mask") is not None:
        masks = [item["uv_mask"] for item in batch]
        max_tp = max(m.shape[0] for m in masks)
        uv_mask = torch.stack([F.pad(m, (0, max_tp - m.shape[0]), value=0.0) for m in masks])

    # phoneme_tokens: [B, Tp] LongTensor
    phoneme_tokens = None
    if batch[0].get("phoneme_tokens") is not None:
        ptoks = [item["phoneme_tokens"] for item in batch]
        max_tp = max(t.shape[0] for t in ptoks)
        phoneme_tokens = torch.stack([F.pad(t, (0, max_tp - t.shape[0]), value=0) for t in ptoks])

    # mu_stars / kappa_stars: [B] float tensors only when ALL samples have cluster targets
    mu_stars = None
    kappa_stars = None
    _mu_vals = [item.get("mu_star") for item in batch]
    if all(v is not None for v in _mu_vals):
        mu_stars = torch.tensor(_mu_vals, dtype=torch.float32)
    _kappa_vals = [item.get("kappa_star") for item in batch]
    if all(v is not None for v in _kappa_vals):
        kappa_stars = torch.tensor(_kappa_vals, dtype=torch.float32)

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        phonemes=phonemes,
        phoneme_tokens=phoneme_tokens,
        mel_80=mel_80,
        uv_mask=uv_mask,
        mu_stars=mu_stars,
        kappa_stars=kappa_stars,
    )
