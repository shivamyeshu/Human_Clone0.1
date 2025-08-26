from gtts import gTTS
import os
import json
from phonemizer.separator import Separator
from phonemizer import phonemize


def synthesize_speech(text, output_audio_path="../clone_package/output_speech.mp3", lang="en"):
    """Generate speech audio using gTTS"""
    tts = gTTS(text=text, lang=lang)
    tts.save(output_audio_path)
    print(f"✅ Saved speech audio to {output_audio_path}")


def extract_phonemes(text, lang="en-us"):
    """
    Extract phonemes without espeak dependency.
    Uses `ipa` backend instead of espeak.
    """
    phonemes = phonemize(
        text,
        language=lang,
        backend="segments",   # pure python backend (no espeak needed)
        strip=True,
        separator=Separator(phone=" ", word=" | ", syllable="")  # spaces between phonemes
    )
    phoneme_list = phonemes.split()
    print(f"✅ Extracted phoneme sequence: {phoneme_list}")
    return phoneme_list


def save_phonemes_to_json(phonemes_list, output_json_path="../clone_package/phonemes.json"):
    """Save phoneme list to JSON"""
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(phonemes_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved phoneme sequence to {output_json_path}")


if __name__ == "__main__":
    # Example usage
    script_text = """so hello whatsup hows u guyz doing this its a base of repliworld stage 2 ."""

    synthesize_speech(script_text)
    phonemes = extract_phonemes(script_text)
    save_phonemes_to_json(phonemes)
