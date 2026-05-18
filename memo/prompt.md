
$\mathcal{E}(\mathbf{x}_0) = \frac{1}{N} \sum \max(0, \mu^* - \mu(\mathbf{x}))$.
$\mathcal{E}(\mathbf{x}_0) = \frac{1}{N} [\sum \max(0, \mu^* - \mu(\mathbf{x})) + \max(0, \kappa^* - \mu(\mathbf{kappa}))]$. 

#### train_first_energyDiT_icl.py (for f5tts)
0. UV_mask
   - self._load_tensor2(data) : uv_mask  (meldataset2.py)
1. Force alignment(FA)
   - model.text_aligner(mels, mask, texts): s2s_attn ...
2. Frame-level pitch extraction
   - model.pitch_extractor(gt.unsqueeze(1)): F0_real
3. Phoneme-level pitch extraction
   - calculate_phoneme_pitch_mean(F0_real, s2s_cutphone): p_f0_mean  
   - Note: Do not need to cut_phn?
4. compute energy_softmax
   - calculate_energy_softmax(p_f0_mean, mu_star, uv_masks_cut, beta=10): energy_softmax
   - Note: temporarily set mu_star to constant value
5. calculate energy_guided_cfm_loss
   - (energy_softmax * loss_per_sample).sum(): loss_cefm


### cefm.py
Create a new cefm.py, a modification version of @cfm.py. I need to add below 5 processes, which have already implemented in @ project.
I list up these 5 processes with their implementation details, including *.py file, function name, input and output so you can find out inside @project. 
I also add "note" to show how to adapt to the new @cefm.py

0. UV_mask
   - convert text to phonemes, and embeds to tensor. ()
   - self._load_tensor2(data) : uv_mask  (meldataset2.py)
1. Force alignment(FA)
   - texts should be phonemes, not character
   - model.text_aligner(mels, mask, texts): s2s_attn ...  (train_first_energyDiT_icl.py)
2. Frame-level pitch extraction
   - model.pitch_extractor(gt.unsqueeze(1)): F0_real      (train_first_energyDiT_icl.py)
3. Phoneme-level pitch extraction
   - calculate_phoneme_pitch_mean(F0_real, s2s_cutphone): p_f0_mean   (train_first_energyDiT_icl.py)
   - Note: Do not need to cut_phn if the f5tts didnot cut mel
4. compute energy_softmax
   - calculate_energy_softmax(p_f0_mean, mu_star, uv_masks_cut, beta=10): energy_softmax (train_first_energyDiT_icl.py)
   - Note: temporarily set mu_star to constant value
5. calculate energy_guided_cfm_loss
   - (energy_softmax * loss_per_sample).sum(): loss_cefm  (flow_matching_v5_energy.py)

If you have question, just ask me

Create a new train_energy.py, a modification of @train.py, which call cefm.py for training.


In EnergyDataset of train_energy.py, I need to do the following modification
1. Convert text to phonemes and output them
   - text is the second item in: row = self.base.data[index]  # audio_path, text, duration
   - phonemes conversion from text can refer to: ps = global_phonemizer.phonemize([text])  (/home/rosen/Project/StyleTTS2/modify_train_list_txt.py)
   - Check if the converted phonemes is correct or not by comparing the phonemes in /home/rosen/Project/StyleTTS2/Data/train_list_libritts_spk.txt.
2. Create uv_mask from phonemes and output them
   - uv_mask creation can refer to: ..., uv_mask = self._load_tensor2(data)  (/home/rosen/Project/StyleTTS2/meldataset2.py)

In cefm.py, 
1. you need to use the self.text_aligner as same as it used in /home/rosen/Project/StyleTTS2/train_first_txt2mel_cfm_icl.py.
   - referring to: ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts) in /home/rosen/Project/StyleTTS2/train_first_txt2mel_cfm_icl.py
   - currently, the text in "ppgs, s2s_pred, s2s_attn = self.text_aligner(mels_80, asr_pad_mask, text)" is character, not phonemes, which is not correct 



### Phoneme-level pitch (extract_phn_pitch)
- Process
  1. Interpolate unvoiced (f0 <= 50) frames of f0.
     - refer to code from "# 1. Linear interpolation as ..." in @cefm_v2.py
  2. Semitone conversion (12 * log2(f0 / 100)).
     - refer to: line 153
  3. Median Filter (Window=3).
     - refer to: the following median_filter
  4. Phoneme-level Aggregation: phoneme-level pitch via the frame2phoneme attention
     - code exist

```python
def median_filter(x: torch.Tensor, window_size: int = 3) -> torch.Tensor:
   """
   Args:
       x: [B, Tf] 序列
       window_size: 滑动窗口大小 (必须是奇数)
   Returns:
       [B, Tf] 滤波后的序列
   """
   B, Tf = x.shape
   padding = window_size // 2
  
   # 1. 填充边界以保持序列长度一致
   # 使用 'replicate' 填充可以防止边缘出现零值抖动
   x_padded = F.pad(x.unsqueeze(1), (padding, padding), mode='replicate') # [B, 1, Tf + 2*padding]
  
   # 2. 使用 unfold 提取滑动窗口
   # [B, 1, Tf, window_size]
   windows = x_padded.unfold(dimension=2, size=window_size, step=1)
  
   # 3. 计算窗口内的中值
   # 注意：torch.median 返回 (values, indices)，我们只需要 values
   x_median = windows.median(dim=-1).values # [B, 1, Tf]
  
   return x_median.squeeze(1)
```


### Phoneme-level pitch curvature (extract_phn_pitch_curv)
- Process
  1. Interpolate unvoiced (f0 <= 50) frames of f0.
     - same as extract_phn_pitch
  2. Semitone conversion (12 * log2(f0 / 100)).
     - same as extract_phn_pitch
  3. Savitzky-Golay smoothing (Window=7).
     - code exist
  4. Second Derivative: Second derivative of smoothed pitch (|d²f0/dt²|)
     - code exist
  5. Phoneme-level Aggregation: phoneme-level pitch curvature via the frame2phoneme attention
     - code exist

### Energy softmax (calculate_energy_softmax)
1. Select phoneme-level pitch (or pitch curvature) of voiced phonemes given uv_mask (generated by phoneme).
2. Select phoneme-level pitch (or pitch curvature) of the phonemes whose value > Q95. (when energy_type != "all_phonemes")
3. Compute energy by the difference between phoneme-level pitch (or pitch curvature) and the corresponding OOD target
4. Z-score normalizataion
5. Apply softmax (over batch) with beta scaled by batch size (so weights sum to B).


### process_single_line in calculate_spk_independent_ood_target_v3.py
1. Interpolate unvoiced (f0 <= 50) frames of f0.
   - refer to code from "# 1. Linear interpolation as ..." in @cefm_v2.py
2. Semitone conversion (12 * log2(f0 / 100)).
   - refer to: line 153
(for phoneme-level pitch)
3a. Median Filter (Window=3).
   - refer to extract_phn_pitch
4a. Phoneme-level Aggregation: phoneme-level pitch via the frame2phoneme attention
   - refer to extract_phn_pitch
(for phoneme-level pitch curvature)
3b. Savitzky-Golay smoothing (Window=7).
   - refer to extract_phn_pitch_curv
4b. Second Derivative: Second derivative of smoothed pitch (|d²f0/dt²|)
   - refer to extract_phn_pitch_curv
5b. Phoneme-level Aggregation: phoneme-level pitch curvature via the frame2phoneme attention
   - refer to extract_phn_pitch_curv