
import phonemizer
from nltk.tokenize import word_tokenize
from text_utils import TextCleaner

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)


def modify_libritts_list():
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            # extract speaker id (the number between train-100-360-500/ and /)
            try:
                spk_id = line.split("train-100-360-500/")[1].split("/")[0]
                new_line = f"{line}|{spk_id}\n"
                f_out.write(new_line)
            except IndexError:
                print(f"Skipped malformed line: {line}")


def modify_esd():
    input_path = "/hdd/ESD/esd.txt"  # your input file
    output_path = "/home/rosen/Project/StyleTTS2/Data/esd_spk.txt"  # your output file
    textclenaer = TextCleaner()
    punctuations = {'.', ',', '?', '!', ';', ':', '"', "'", '…'}

    # initialize phonemizer (English in this example)
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                wav_path, text = line.split("|", 1)
                # step 1: phonemize text
                ps = global_phonemizer.phonemize([text])
                ps = word_tokenize(ps[0].strip())
                # merge puncturation
                merged = []
                for token in ps:
                    if token in punctuations and merged:
                        merged[-1] += token  # append punctuation to previous token
                    else:
                        merged.append(token)

                ps = ' '.join(merged)

                # step 2: extract speaker id between /hdd/ESD/ and /
                spk_id = wav_path.split("/hdd/ESD/")[1].split("/")[0]

                # step 3: write to new file
                new_line = f"{wav_path}|{ps}|{spk_id}\n"
                f_out.write(new_line)
            except Exception as e:
                print(f"Error processing line: {line}\n{e}")


if __name__ == '__main__':
    input_path = "/home/rosen/Project/StyleTTS2/Data/val_list_libritts.txt"  # your input file path
    output_path = "/home/rosen/Project/StyleTTS2/Data/val_list_libritts_spk.txt"  # your output file path
    modify_esd()
