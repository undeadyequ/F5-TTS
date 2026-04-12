import sys
import json
import numpy as np
import librosa
import scipy.signal
import scipy.interpolate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.cluster import KMeans

sys.path.insert(0, '/home/rosen/Project/energyDiT')

# Voiced IPA symbols (matches build_voiced_mask in meldataset2.py)
VOICED_PHONES = {
    # vowels
    "i", "y", "ɨ", "ʉ", "ɯ", "u",
    "ɪ", "ʏ", "ʊ",
    "e", "ø", "ɘ", "ɵ", "ɤ", "o",
    "ə", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ",
    "æ", "ɐ", "a", "ɶ", "ɑ", "ɒ",
    # voiced consonants
    "b", "d", "ɡ", "v", "ð", "z", "ʒ",
    "ʝ", "ɣ", "ʁ", "ʕ", "ɦ",
    "m", "n", "ŋ", "ɱ", "ɳ", "ɲ",
    "l", "ɫ", "ɭ", "ʎ", "r", "ɹ", "ɻ", "ɾ",
    "w", "j",
}

# Mel hop length used for FA boundaries (must match fa_utils.MEL_HOP_LENGTH)
# pyin uses the same hop_length so frame indices are directly comparable
FA_HOP_LENGTH = 300
JDC_PATH = "/home/rosen/Project/energyDiT/Utils/JDC/bst.t7"
STYLETTS2_ROOT = "/home/rosen/Project/StyleTTS2"

# Module-level cache so the pitch extractor is loaded once per worker process
_pitch_extractor = None
_pitch_extractor_device = None


def _get_pitch_extractor(device):
    global _pitch_extractor, _pitch_extractor_device
    if _pitch_extractor is None or _pitch_extractor_device != device:
        import torch
        if STYLETTS2_ROOT not in sys.path:
            sys.path.insert(0, STYLETTS2_ROOT)
        from models import load_F0_models
        _pitch_extractor = load_F0_models(JDC_PATH).to(device).eval()
        _pitch_extractor_device = device
    return _pitch_extractor


def _load_mel_cpu(wav_path):
    """Thread worker: load wav and compute mel on CPU. Returns (wav_path, mel_np) or None."""
    try:
        import torchaudio
        import torch
        wave, _ = librosa.load(wav_path, sr=24000)
        wave_t = torch.from_numpy(wave).float()
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=FA_HOP_LENGTH
        )
        mel = to_mel(wave_t)                                      # [n_mels, T]
        mel = (torch.log(1e-5 + mel.unsqueeze(0)) - (-4)) / 4    # [1, n_mels, T]
        return wav_path, mel.numpy()
    except Exception:
        return wav_path, None


def extract_f0_jdc_batch(wav_paths, device="cuda", batch_size=32, io_workers=4,
                         load_chunk=1000):
    """Extract F0 (Hz) for a list of wav files using the JDC pitch extractor.

    Processes files in chunks of `load_chunk` to bound RAM — only one chunk of
    mels is resident at a time. GPU inference runs on sub-batches of `batch_size`
    within each chunk.

    Args:
        wav_paths:   list of wav file paths
        device:      torch device string
        batch_size:  number of mels per GPU forward pass
        io_workers:  threads for parallel mel loading per chunk
        load_chunk:  number of wav files loaded into RAM at once (~1.6GB per 1000)

    Returns:
        dict {wav_path: np.ndarray}  — F0 in Hz, shape [T_mel], 0 = unvoiced
    """
    import torch
    from concurrent.futures import ThreadPoolExecutor

    pitch_extractor = _get_pitch_extractor(device)
    results = {}
    n_chunks = (len(wav_paths) + load_chunk - 1) // load_chunk

    for chunk_start in tqdm(range(0, len(wav_paths), load_chunk),
                            total=n_chunks, desc="JDC F0"):
        chunk_paths = wav_paths[chunk_start: chunk_start + load_chunk]

        # ── 1. Load mels for this chunk in parallel threads ────────────────
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            loaded = list(pool.map(_load_mel_cpu, chunk_paths))

        # ── 2. GPU inference in batch_size sub-batches ─────────────────────
        for start in range(0, len(loaded), batch_size):
            sub = loaded[start: start + batch_size]
            valid = [(p, m) for p, m in sub if m is not None]
            if not valid:
                continue

            paths_v, mels_v = zip(*valid)
            mel_lengths = [m.shape[-1] for m in mels_v]
            max_len = max(mel_lengths)

            padded = torch.zeros(len(mels_v), 1, mels_v[0].shape[1], max_len)
            for k, m in enumerate(mels_v):
                padded[k, 0, :, : m.shape[-1]] = torch.from_numpy(m[0])
            padded = padded.to(device)

            with torch.no_grad():
                f0_gpu, _, _ = pitch_extractor(padded)   # [B, T]

            for k, (path, orig_len) in enumerate(zip(paths_v, mel_lengths)):
                results[path] = f0_gpu[k, :orig_len].cpu().numpy()

        del loaded   # free mels before loading next chunk

    return results


def compute_fa_train(input_file, output_fa_json, max_lines=10000, batch_size=32, io_workers=4):
    """Run phoneme forced alignment on training data and save to fa_train.json.

    Uses batched GPU inference + parallel audio I/O for speed.

    Args:
        input_file:      path to LibriTTS data file (path|phones_str|spk_id per line)
        output_fa_json:  path to save FA results
        max_lines:       maximum number of lines to process
        batch_size:      number of samples per GPU batch (tune to VRAM)
        io_workers:      threads for parallel audio loading within each batch

    FA result format:
        {wav_id: {"phones": [chars...], "boundaries": [mel_frame_indices...]}}
    """
    from utilities.fa_utils import load_fa_models, run_fa_batch
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FA models on {device}...")
    text_aligner, text_cleaner, to_mel, mel_mean, mel_std = load_fa_models(device)

    with open(input_file) as f:
        lines = f.readlines()
    lines = lines[:max_lines]
    print(f"Running FA for {len(lines)} files (batch_size={batch_size}, io_workers={io_workers})...")

    # Build (wav_path, phones_str, wav_id) list
    samples = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2:
            continue
        samples.append((parts[0], parts[1], Path(parts[0]).stem))

    fa_data = {}
    for start in tqdm(range(0, len(samples), batch_size), desc="FA batches"):
        chunk   = samples[start: start + batch_size]
        batch   = [(wav_path, phones_str) for wav_path, phones_str, _ in chunk]
        results = run_fa_batch(batch, text_aligner, text_cleaner, device,
                               to_mel, mel_mean, mel_std, io_workers=io_workers)
        for (_, _, wav_id), (phones, boundaries) in zip(chunk, results):
            if phones is None:
                continue
            fa_data[wav_id] = {"phones": phones, "boundaries": boundaries}

    with open(output_fa_json, 'w') as f:
        json.dump(fa_data, f, indent=4)
    print(f"FA saved to {output_fa_json} ({len(fa_data)} entries)")
    return fa_data


def process_single_line(args):
    """Worker function to process one line and return voiced phoneme stats.

    Args:
        args: tuple of (line, fa_entry, f0_hz) where:
              - fa_entry: {"phones": [...], "boundaries": [...]}
              - f0_hz: pre-computed F0 array in Hz (np.ndarray, 0=unvoiced), or None to use pyin
    """
    line, fa_entry, f0_hz = args
    try:
        path_str, phones_str, speaker = line.strip().split('|')
        wav_id = Path(path_str).stem

        # phones: individual IPA characters including spaces (matches FA alignment)
        phones = list(phones_str)

        if f0_hz is not None:
            voiced_flag = f0_hz > 0
            f0 = np.where(voiced_flag, f0_hz, np.nan)
        else:
            y, _ = librosa.load(path_str, sr=24000)
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=24000,
                hop_length=FA_HOP_LENGTH,
            )

        # Log-scale conversion for human-like perception
        f0_semitones = np.where(voiced_flag, 12 * np.log2(np.where(voiced_flag, f0, 100.0) / 100.0), np.nan)
        # Smooth using cubic spline interpolation and Savitzky-Golay
        valid_idx = np.where(~np.isnan(f0_semitones))[0]

        #valid_idx = np.where(voiced_flag)[0] # Cubic spline interpolation of unvoiced gaps (in Hz), then Savitzky-Golay
        if len(valid_idx) < 2:
            return wav_id, {}

        interp_func = scipy.interpolate.interp1d(
            valid_idx, f0[valid_idx], kind='cubic', fill_value="extrapolate")
        f0_smoothed = scipy.signal.savgol_filter(
            interp_func(np.arange(len(f0))), window_length=7, polyorder=2)

        # Determine phoneme boundaries
        if fa_entry is not None:
            fa_phones = fa_entry["phones"]
            boundaries = fa_entry["boundaries"]
            # Prefer FA phones (same source as text_cleaner), fall back to parsed phones
            if len(fa_phones) == len(phones):
                phones = fa_phones
        else:
            raise IOError("fa_entry should not be none")

        wav_phone_stats = {}
        n_f0 = len(f0_smoothed)

        mus = []
        d2s = []
        for i, phone in enumerate(phones):
            # Only process voiced phonemes
            if phone not in VOICED_PHONES:
                continue

            start = int(boundaries[i]) if i < len(boundaries) else n_f0
            end = int(boundaries[i + 1]) if i + 1 < len(boundaries) else n_f0
            start = min(start, n_f0)
            end = min(end, n_f0)
            if end - start < 2:
                continue

            # nanmean: ignore unvoiced frames (NaN) within a voiced phoneme span
            mu = np.nanmean(f0_semitones[start:end])
            if np.isnan(mu):
                continue
            # Curvature (Second Derivative) on smoothed f0
            d2 = np.gradient(np.gradient(f0_smoothed[start:end]))
            kappa = np.nanmean(np.abs(d2))
            mus.append(mu)
            d2s.append(kappa)
        return wav_id, mus, d2s, speaker
    except Exception:
        return None, None


OUTPUT_DIR = Path("/home/rosen/Project/energyDiT/Data_ood")


def main(input_file, num_workers=12, fa_json=None, compute_fa=False, max_lines=10000,
         speakers_txt="/home/rosen/data/LibriTTS_f5tts/SPEAKERS.txt",
         fa_batch_size=32, fa_io_workers=4,
         mu_path=None, kappa_path=None, use_jdc=False,
         jdc_batch_size=32, jdc_load_chunk=1000, cluster_by="both",
         start_step=0, end_step=4):
    """Main entry point.

    Steps:
        0 - Forced alignment (FA)
        1 - Extract phoneme-level mu (log F0 mean) and kappa (curvature)
        2 - Compute Q55/Q75/Q95 per speaker
        3 - K-means cluster speakers (M/F separately)
        4 - Record Q55/Q75/Q95 per cluster group

    Args:
        input_file: path to LibriTTS data file
        num_workers: parallel workers for feature extraction
        fa_json: path to FA json file (will be created if compute_fa=True)
        compute_fa: if True, run FA alignment and save fa_train.json
        max_lines: max lines to process
        start_step: first step to run (inclusive, 0–4)
        end_step: last step to run (inclusive, 0–4)
    """
    suffix = f"_{max_lines}" if max_lines != -1 else ""
    cluster_suffix = f"_{cluster_by}" if cluster_by != "both" else ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if fa_json is None:
        fa_json = str(OUTPUT_DIR / f'fa_train{suffix}.json')
    if mu_path is None:
        mu_path = str(OUTPUT_DIR / f'wav_phone_mu{suffix}.txt')
    if kappa_path is None:
        kappa_path = str(OUTPUT_DIR / f'wav_phone_kappa{suffix}.txt')
    q_path         = str(OUTPUT_DIR / f'spk_pf0meancurv_q55_75_95{suffix}.json')
    cluster_path   = str(OUTPUT_DIR / f'spk_clusters{suffix}{cluster_suffix}.json')
    cluster_q_path = str(OUTPUT_DIR / f'spk_clusters_q55_q75_q95{suffix}{cluster_suffix}.json')

    # ── Step 0: Forced Alignment ───────────────────────────────────────────────
    fa_data = {}
    if start_step <= 1 <= end_step:  # fa_data is needed only for step 1
        if start_step <= 0 and compute_fa:
            fa_data = compute_fa_train(input_file, fa_json, max_lines=max_lines,
                                       batch_size=fa_batch_size, io_workers=fa_io_workers)
        elif Path(fa_json).exists():
            print(f"Loading existing FA data from {fa_json}...")
            with open(fa_json) as f:
                fa_data = json.load(f)
            print(f"  Loaded {len(fa_data)} entries")
        else:
            raise IOError(f"No FA data found at {fa_json}. Run with --compute-fa or provide --fa-json.")
    elif start_step == 0 <= end_step and compute_fa:
        # Step 0 explicitly requested but step 1 not in range — still run FA
        fa_data = compute_fa_train(input_file, fa_json, max_lines=max_lines,
                                   batch_size=fa_batch_size, io_workers=fa_io_workers)

    # ── Step 1: Extract phoneme-level mu and kappa ─────────────────────────────
    if start_step <= 1 <= end_step:
        if Path(mu_path).exists() and Path(kappa_path).exists():
            print(f"Skipping Step 1: using existing {mu_path} and {kappa_path}")
        else:
            with open(input_file, 'r') as f:
                lines = f.readlines()
            lines = lines[:max_lines]

            def get_fa_entry(line):
                parts = line.strip().split('|')
                if not parts:
                    return None
                wav_id = Path(parts[0]).stem
                return fa_data.get(wav_id, None)

            # Pre-compute F0 via JDC batch GPU inference if requested
            jdc_f0_map = {}
            if use_jdc:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                wav_paths = [line.strip().split('|')[0] for line in lines
                             if line.strip().split('|')]
                print(f"Extracting F0 via JDC on {device} for {len(wav_paths)} files...")
                jdc_f0_map = extract_f0_jdc_batch(wav_paths, device=device,
                                                  batch_size=jdc_batch_size,
                                                  load_chunk=jdc_load_chunk)

            args_list = [(line, get_fa_entry(line), jdc_f0_map.get(line.strip().split('|')[0]))
                         for line in lines]  # lines, fa, pitch

            print(f"Processing {len(lines)} files...")
            if num_workers > 1:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(process_single_line, args_list),
                                        total=len(args_list)))
            else:
                results = list(map(process_single_line, args_list))
            print("Step 1 done!")

            with open(mu_path, 'w') as fm, open(kappa_path, 'w') as fk:
                for row in results:
                    if len(row) != 4 or row[0] is None:
                        continue
                    wav_id, mus, kappas, spk = row
                    if not mus:
                        continue
                    fm.write(wav_id + "|" + spk + "|" + ",".join(f"{v:.2f}" for v in mus) + "\n")
                    fk.write(wav_id + "|" + spk + "|" + ",".join(f"{v:.2f}" for v in kappas) + "\n")
            print(f"Saved {mu_path} and {kappa_path}")

    # ── Step 2: Q55/Q75/Q95 per speaker ───────────────────────────────────────
    spk_quantiles = {}
    if start_step <= 2 <= end_step:
        def _load_spk_vals(path):
            spk_dict = {}
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    if len(parts) != 3:
                        continue
                    spk = parts[1]
                    vals = [float(v) for v in parts[2].split(",") if v]
                    spk_dict.setdefault(spk, []).extend(vals)
            return spk_dict

        spk_mus    = _load_spk_vals(mu_path)
        spk_kappas = _load_spk_vals(kappa_path)

        quantiles = [0.55, 0.75, 0.95]
        for spk in spk_mus:
            mu_arr = np.array(spk_mus[spk])
            k_arr  = np.array(spk_kappas.get(spk, [0.0]))
            mu_qs  = np.quantile(mu_arr, quantiles).tolist()
            k_qs   = np.quantile(k_arr,  quantiles).tolist()
            spk_quantiles[spk] = [mu_qs, k_qs]

        with open(q_path, 'w') as f:
            json.dump(spk_quantiles, f, indent=4)
        print(f"Saved per-speaker quantiles to {q_path} ({len(spk_quantiles)} speakers)")
    elif start_step <= 3 <= end_step or start_step <= 4 <= end_step:
        print(f"Loading existing per-speaker quantiles from {q_path}...")
        with open(q_path) as f:
            spk_quantiles = json.load(f)
        print(f"  Loaded {len(spk_quantiles)} speakers")

    # ── Step 3: Cluster speakers ───────────────────────────────────────────────
    cluster_results = {}
    if start_step <= 3 <= end_step:
        cluster_results = cluster_speakers(spk_quantiles, suffix, speakers_txt=speakers_txt,
                                           cluster_by=cluster_by)
        with open(cluster_path, 'w') as f:
            json.dump(cluster_results, f, indent=4)
        print(f"Saved speaker clusters to {cluster_path}")
    elif start_step <= 4 <= end_step:
        print(f"Loading existing speaker clusters from {cluster_path}...")
        with open(cluster_path) as f:
            cluster_results = json.load(f)
        print(f"  Loaded {len(cluster_results)} groups")

    # ── Step 4: Q55/Q75/Q95 per cluster group ─────────────────────────────────
    if start_step <= 4 <= end_step:
        cluster_quantiles = {}
        for group_name, (_, members) in cluster_results.items():
            mu_q55s, mu_q75s, mu_q95s = [], [], []
            k_q55s,  k_q75s,  k_q95s  = [], [], []
            for spk in members:
                if spk not in spk_quantiles:
                    continue
                mu_qs, k_qs = spk_quantiles[spk]
                mu_q55s.append(mu_qs[0]); mu_q75s.append(mu_qs[1]); mu_q95s.append(mu_qs[2])
                k_q55s.append(k_qs[0]);   k_q75s.append(k_qs[1]);   k_q95s.append(k_qs[2])
            cluster_quantiles[group_name] = {
                "mu":    [float(np.median(mu_q55s)), float(np.median(mu_q75s)), float(np.median(mu_q95s))],
                "kappa": [float(np.median(k_q55s)),  float(np.median(k_q75s)),  float(np.median(k_q95s))],
            }
        with open(cluster_q_path, 'w') as f:
            json.dump(cluster_quantiles, f, indent=4)
        print(f"Saved cluster quantiles to {cluster_q_path}")

def load_speaker_gender(speakers_txt="/home/rosen/data/LibriTTS_f5tts/SPEAKERS.txt"):
    """Parse LibriTTS SPEAKERS.txt and return {spk_id: 'M'|'F'}."""
    gender_map = {}
    with open(speakers_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            try:
                spk_id = str(int(parts[0]))
                gender = parts[1].strip().upper()
                if gender in ("M", "F"):
                    gender_map[spk_id] = gender
            except ValueError:
                continue
    return gender_map


def cluster_speakers(spk_quantiles, suffix="", n_clusters=3,
                     speakers_txt="/home/rosen/data/LibriTTS_f5tts/SPEAKERS.txt",
                     cluster_by="both"):
    """K-means cluster speakers (separately for M and F) by Q95 of mu and/or kappa.

    Args:
        spk_quantiles: {spk_id: [[mu_Q55, mu_Q75, mu_Q95], [k_Q55, k_Q75, k_Q95]]}
        suffix: suffix for output plot/file names
        n_clusters: number of clusters per gender (default 3 → low/mid/high)
        speakers_txt: path to SPEAKERS.txt
        cluster_by: "both" (default) | "mu" | "kappa"
            "both"  — feature = [mu_Q95, kappa_Q95], sort by mu centroid
            "mu"    — feature = [mu_Q95],             sort by mu centroid
            "kappa" — feature = [kappa_Q95],          sort by kappa centroid

    Returns:
        dict with keys like "M_low", "M_mid", "M_high", "F_low", "F_mid", "F_high"
        Each value: [[mu_Q95_median, kappa_Q95_median], [spk1, spk2, ...]]
    """
    assert cluster_by in ("both", "mu", "kappa"), \
        f"cluster_by must be 'both', 'mu', or 'kappa', got '{cluster_by}'"

    gender_map = load_speaker_gender(speakers_txt)
    label_map = {0: "low", 1: "mid", 2: "high"}

    result = {}
    for gender in ("M", "F"):
        spks = [s for s in spk_quantiles if gender_map.get(s, "?") == gender]
        # Drop speakers with NaN quantiles
        spks = [s for s in spks if not any(
            np.isnan(v) for v in spk_quantiles[s][0] + spk_quantiles[s][1]
        )]
        if len(spks) < n_clusters:
            print(f"Warning: only {len(spks)} valid {gender} speakers, skipping clustering.")
            continue

        # Build feature matrix and sort-axis index based on cluster_by
        if cluster_by == "mu":
            X = np.array([[spk_quantiles[s][0][2]] for s in spks])   # [N, 1]
            sort_axis = 0
        elif cluster_by == "kappa":
            X = np.array([[spk_quantiles[s][1][2]] for s in spks])   # [N, 1]
            sort_axis = 0
        else:  # both
            X = np.array([[spk_quantiles[s][0][2], spk_quantiles[s][1][2]] for s in spks])
            sort_axis = 0  # sort by mu (first column)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Sort cluster indices by centroid of sort_axis ascending → low/mid/high
        centroid_sort = km.cluster_centers_[:, sort_axis]
        rank = np.argsort(centroid_sort)
        remap = {orig: new for new, orig in enumerate(rank)}

        clusters = {i: [] for i in range(n_clusters)}
        for spk, lbl in zip(spks, labels):
            clusters[remap[lbl]].append(spk)

        for i in range(n_clusters):
            key = f"{gender}_{label_map[i]}"
            members = clusters[i]
            mu_med    = float(np.median([spk_quantiles[s][0][2] for s in members]))
            kappa_med = float(np.median([spk_quantiles[s][1][2] for s in members]))
            result[key] = [[mu_med, kappa_med], members]

        _plot_clusters(spks, X, labels, remap, label_map, gender, suffix, cluster_by)

    return result


def _plot_clusters(spks, X, labels, remap, label_map, gender, suffix, cluster_by="both"):
    """Scatter plot coloured by cluster. 2D when cluster_by='both', 1D strip otherwise."""
    colors = ["steelblue", "darkorange", "forestgreen"]
    plt.figure(figsize=(7, 5))
    n_features = X.shape[1]
    xlabel = "μ Q95 (semitones)" if cluster_by in ("both", "mu") else "κ Q95"
    for orig_lbl, ranked_lbl in remap.items():
        mask = labels == orig_lbl
        x_vals = X[mask, 0]
        if n_features > 1:
            y_vals = X[mask, 1]
        else:
            # 1D clustering: jitter along y so points don't all overlap
            y_vals = np.random.default_rng(ranked_lbl).uniform(-0.1, 0.1, size=mask.sum())
        plt.scatter(x_vals, y_vals,
                    c=colors[ranked_lbl], label=label_map[ranked_lbl], alpha=0.7, s=30)
    plt.xlabel(xlabel)
    plt.ylabel("κ Q95" if n_features > 1 else "")
    cluster_suffix = f"_{cluster_by}" if cluster_by != "both" else ""
    plt.title(f"{gender} speaker clusters by {cluster_by} (n={len(spks)})")
    plt.legend(title="cluster")
    plt.tight_layout()
    path = str(OUTPUT_DIR / f"spk_clusters_{gender}{suffix}{cluster_suffix}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved cluster plot: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute phoneme-level F0_mean and F0_curvature")
    parser.add_argument("--input", default="/home/rosen/Project/energyDiT/Data/train_libritts_100_360_500.txt",
                        help="LibriTTS data file (path|phones_str|spk_id)")
    parser.add_argument("--fa-json", default=None,
                        help="Path to FA json file (default: same dir as input / fa_train.json)")
    parser.add_argument("--compute-fa", action="store_true",
                        help="Run forced alignment and save fa_train.json before feature extraction")
    parser.add_argument("--fa-batch-size", type=int, default=32,
                        help="GPU batch size for FA inference (default: 32)")
    parser.add_argument("--fa-io-workers", type=int, default=4,
                        help="Threads for parallel audio loading during FA (default: 4)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Parallel workers for feature extraction (default: 8)")
    parser.add_argument("--max-lines", type=int, default=10000,
                        help="Maximum number of lines to process (default: 10000)")
    parser.add_argument("--speakers-txt",
                        default="/home/rosen/data/LibriTTS_f5tts/SPEAKERS.txt",
                        help="Path to LibriTTS SPEAKERS.txt for gender lookup")
    parser.add_argument("--mu-path", default=None,
                        help="Path to existing wav_phone_mu.txt (skips Step 1 if both given)")
    parser.add_argument("--kappa-path", default=None,
                        help="Path to existing wav_phone_kappa.txt (skips Step 1 if both given)")
    parser.add_argument("--use-jdc", action="store_true",
                        help="Use JDC pitch extractor instead of librosa.pyin")
    parser.add_argument("--jdc-batch-size", type=int, default=32,
                        help="GPU batch size for JDC inference (default: 32)")
    parser.add_argument("--jdc-load-chunk", type=int, default=1000,
                        help="Files loaded into RAM per chunk for JDC (default: 1000)")
    parser.add_argument("--cluster-by", default="both", choices=["both", "mu", "kappa"],
                        help="Feature(s) to use for K-means clustering: 'both' (default), 'mu', or 'kappa'")
    parser.add_argument("--start-step", type=int, default=0, choices=range(5),
                        help="First step to run (0-4, default: 0)")
    parser.add_argument("--end-step", type=int, default=4, choices=range(5),
                        help="Last step to run (0-4, default: 4)")
    args = parser.parse_args()

    main(
        input_file=args.input,
        num_workers=args.num_workers,
        fa_json=args.fa_json,
        compute_fa=args.compute_fa,
        max_lines=args.max_lines,
        speakers_txt=args.speakers_txt,
        fa_batch_size=args.fa_batch_size,
        fa_io_workers=args.fa_io_workers,
        mu_path=args.mu_path,
        kappa_path=args.kappa_path,
        use_jdc=args.use_jdc,
        jdc_batch_size=args.jdc_batch_size,
        jdc_load_chunk=args.jdc_load_chunk,
        cluster_by=args.cluster_by,
        start_step=args.start_step,
        end_step=args.end_step,
    )
