f5-tts_infer-cli \
  --model F5TTS_v1_Small \
  --ckpt_file ckpts/F5TTS_v1_Small_vocos_char_LibriTTS_100_360_500/model_last.pt \
  --vocab_file data/LibriTTS_100_360_500_char/vocab.txt \
  --ref_audio "/home/rosen/Project/F5-TTS/res/0019_001402.wav" \
  --ref_text "I did go, and made many prisoners." \
  --gen_text "Some text you want TTS model generate for you."



"I did go, and made many prisoners."