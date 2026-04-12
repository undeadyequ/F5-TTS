
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