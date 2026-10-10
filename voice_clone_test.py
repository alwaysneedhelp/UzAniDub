"""
voice_clone_test.py

A minimal, standalone test harness for one specific question:

    Can we take a video, extract one character's voice, and clone that
    voice onto a Uzbek TTS output well enough that it's recognizable?

This is NOT the full dubbing pipeline. It's a focused experiment so you
can hear the answer before building anything around it.

Pipeline (all local, no cloud APIs):
  1. ffmpeg: extract the full audio track from the video
  2. Demucs (optional): separate vocals from music/SFX
  3. You: tell the script the start/end (seconds) of a clean, single-speaker
     line -- this becomes the voice reference
  4. facebook/mms-tts-uzb-script_cyrillic: generate correct Uzbek speech for
     a test sentence (generic voice, correct pronunciation)
  5. OpenVoice V2 tone color converter: re-voice that Uzbek audio into the
     reference speaker's timbre

READ THE CRITICAL NOTES BELOW before judging the result.

Usage:
    python voice_clone_test.py \
        --video "Naruto scene.mp4" \
        --ref-start 12.0 --ref-end 18.5 \
        --text "Salom, bugun ajoyib kun, sen buni uddalaysan."

Outputs (written under ./work/ next to wherever you run this):
    work/audio_full.wav        full extracted audio
    work/demucs_out/...        Demucs-separated vocals (if not skipped)
    work/reference.wav         your sliced reference clip
    work/uzbek_generic.wav     MMS Uzbek TTS output, generic voice
    work/uzbek_cloned.wav      final result: Uzbek content, cloned voice

CRITICAL NOTES -- read before judging the output:
  - facebook/mms-tts-uzb-script_cyrillic is licensed CC-BY-NC 4.0
    (non-commercial only). Fine to use for this local test. NOT fine to
    ship in anything meant to be freely/commercially usable without
    swapping it for a differently-licensed model later.
  - It only accepts Cyrillic-script Uzbek text. This script includes a
    crude Latin->Cyrillic transliterator (see latin_to_cyrillic_uz below)
    so you can type normally, but it's approximate -- good enough for a
    listening test, not for production.
  - OpenVoice's cross-lingual claim is real and documented (MIT license,
    genuinely zero-shot, no per-speaker training needed), but it was
    validated on English/Spanish/French/Chinese/Japanese/Korean. Uzbek's
    vowel harmony and sounds like o'/g'/q/h are untested territory for it.
    This script exists to find out empirically, not to confirm a foregone
    conclusion -- go in expecting it might sound wrong or strained.
  - Anime dialogue (shouting, big pitch swings, music/SFX bleeding through
    even after Demucs) is a harder case than the calm narrated-audiobook
    clips most cloning demos use. Don't be surprised if this sounds worse
    than demo videos you've seen online.
  - This tests ONE short line from ONE speaker. It says nothing about
    consistency across a full scene, emotional range, or multiple
    overlapping speakers.
  - RVC / so-vits-svc are deliberately NOT used here: they need up to
    ~10 minutes of clean per-speaker audio to train a usable model, which
    one scene won't give you. They only make sense if you later build a
    proper training set for one recurring character across many episodes.

Setup: see SETUP.md before running this.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

WORK_DIR = Path("work")


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def extract_audio(video_path: Path, out_path: Path):
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(out_path),
    ])


def separate_vocals(audio_path: Path, out_dir: Path) -> Path:
    """Runs Demucs and returns the path to the isolated vocals stem."""
    run([
        sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals",
        "-o", str(out_dir), str(audio_path),
    ])
    stem = audio_path.stem
    vocals_path = out_dir / "htdemucs" / stem / "vocals.wav"
    if not vocals_path.exists():
        raise FileNotFoundError(
            f"Expected Demucs output at {vocals_path}, but it wasn't created. "
            "Check the Demucs log above for errors."
        )
    return vocals_path


def slice_clip(src: Path, start: float, end: float, out_path: Path):
    duration = end - start
    if duration <= 0:
        sys.exit("--ref-end must be greater than --ref-start")
    run([
        "ffmpeg", "-y", "-i", str(src),
        "-ss", str(start), "-t", str(duration),
        "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
        str(out_path),
    ])


def generate_uzbek_tts(text: str, out_path: Path):
    import torch
    from transformers import VitsModel, AutoTokenizer
    import soundfile as sf

    model_id = "facebook/mms-tts-uzb-script_cyrillic"
    print(f"Loading {model_id} (first run will download it, ~150MB) ...")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    sf.write(str(out_path), output.squeeze().numpy(), model.config.sampling_rate)
    print(f"Wrote generic Uzbek TTS to {out_path}")


# --- crude Latin -> Cyrillic transliteration for Uzbek ------------------
# Approximate only. Longest sequences must be matched first. For anything
# beyond a quick listening test, use a proper library or write the
# sentence directly in Cyrillic and pass --text-cyrillic.
_UZ_MULTI = [
    ("sh", "ш"), ("Sh", "Ш"), ("SH", "Ш"),
    ("ch", "ч"), ("Ch", "Ч"), ("CH", "Ч"),
    ("yo", "ё"), ("Yo", "Ё"),
    ("yu", "ю"), ("Yu", "Ю"),
    ("ya", "я"), ("Ya", "Я"),
    ("o'", "ў"), ("O'", "Ў"), ("g'", "ғ"), ("G'", "Ғ"),
    ("ng", "нг"),
]
_UZ_SINGLE = {
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "ҳ",
    "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "x": "х", "y": "й", "z": "з",
    "A": "А", "B": "Б", "D": "Д", "E": "Е", "F": "Ф", "G": "Г", "H": "Ҳ",
    "I": "И", "J": "Ж", "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О",
    "P": "П", "Q": "Қ", "R": "Р", "S": "С", "T": "Т", "U": "У", "V": "В",
    "X": "Х", "Y": "Й", "Z": "З",
}


def latin_to_cyrillic_uz(text: str) -> str:
    for latin, cyr in _UZ_MULTI:
        text = text.replace(latin, cyr)
    return "".join(_UZ_SINGLE.get(ch, ch) for ch in text)


def clone_voice(source_audio: Path, reference_audio: Path, out_path: Path,
                 device: str = None):
    """
    Uses OpenVoice V2's tone color converter to re-voice `source_audio`
    (correct Uzbek content/timing) into the timbre of `reference_audio`
    (the target speaker).

    Requires the myshell-ai/OpenVoice repo cloned locally with its V2
    checkpoints downloaded -- see SETUP.md. Either run this script from
    inside that repo's root, or set OPENVOICE_REPO to its path.
    """
    openvoice_repo = os.environ.get("OPENVOICE_REPO")
    if openvoice_repo:
        sys.path.insert(0, openvoice_repo)

    import torch
    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
    except ImportError as e:
        raise SystemExit(
            "Could not import openvoice. This script needs the official "
            "myshell-ai/OpenVoice repo available on PYTHONPATH, with its V2 "
            "checkpoints downloaded. Either run this script from inside "
            "that repo's root directory, or set the OPENVOICE_REPO "
            "environment variable to its path. See SETUP.md.\n"
            f"Original error: {e}"
        )

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(os.environ.get("OPENVOICE_CKPT_V2", "checkpoints_v2/converter"))
    if not (ckpt_dir / "config.json").exists():
        raise SystemExit(
            f"Expected OpenVoice V2 converter checkpoint at {ckpt_dir}, "
            "not found. Set the OPENVOICE_CKPT_V2 env var to the correct "
            "path (the 'converter' folder from the V2 checkpoint download), "
            "or see SETUP.md."
        )

    print(f"Using device: {device}")
    converter = ToneColorConverter(str(ckpt_dir / "config.json"), device=device)
    converter.load_ckpt(str(ckpt_dir / "checkpoint.pth"))

    print("Extracting source (generic TTS) speaker embedding...")
    source_se, _ = se_extractor.get_se(str(source_audio), converter, vad=True)
    print("Extracting target (reference) speaker embedding...")
    target_se, _ = se_extractor.get_se(str(reference_audio), converter, vad=True)

    converter.convert(
        audio_src_path=str(source_audio),
        src_se=source_se,
        tgt_se=target_se,
        output_path=str(out_path),
    )
    print(f"Wrote cloned Uzbek output to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help='Path to the video, e.g. "Naruto scene.mp4"')
    parser.add_argument("--ref-start", type=float, required=True, help="Reference clip start (seconds)")
    parser.add_argument("--ref-end", type=float, required=True, help="Reference clip end (seconds)")
    parser.add_argument("--text", required=True, help="Uzbek test sentence (Latin script unless --text-cyrillic)")
    parser.add_argument("--text-cyrillic", action="store_true", help="Treat --text as already Cyrillic")
    parser.add_argument(
        "--skip-separation", action="store_true",
        help="Skip Demucs and slice the reference straight from the raw extracted "
             "audio (faster, but background music/SFX will bleed into the voice "
             "reference and likely hurt cloning quality)",
    )
    parser.add_argument("--device", default=None, help='"cuda:0" or "cpu" (auto-detected if omitted)')
    args = parser.parse_args()

    WORK_DIR.mkdir(exist_ok=True)
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    full_audio = WORK_DIR / "audio_full.wav"
    extract_audio(video_path, full_audio)

    if args.skip_separation:
        ref_source = full_audio
    else:
        print("Running Demucs to separate vocals from music/SFX (can take a few minutes on CPU)...")
        ref_source = separate_vocals(full_audio, WORK_DIR / "demucs_out")

    reference_clip = WORK_DIR / "reference.wav"
    slice_clip(ref_source, args.ref_start, args.ref_end, reference_clip)

    cyrillic_text = args.text if args.text_cyrillic else latin_to_cyrillic_uz(args.text)
    print(f"Uzbek text (Cyrillic, as fed to the model): {cyrillic_text}")

    generic_tts = WORK_DIR / "uzbek_generic.wav"
    generate_uzbek_tts(cyrillic_text, generic_tts)

    cloned_out = WORK_DIR / "uzbek_cloned.wav"
    clone_voice(generic_tts, reference_clip, cloned_out, device=args.device)

    print("\nDone. Listen to these three files in this order:")
    print(f"  1. {reference_clip}   <- the voice you're trying to match")
    print(f"  2. {generic_tts}      <- correct Uzbek, generic voice")
    print(f"  3. {cloned_out}       <- correct Uzbek, attempted cloned voice")
    print("\nJudge #3 against #1 for TIMBRE similarity, not against your memory")
    print("of the character's English/Japanese voice acting -- you're checking")
    print("whether the voice quality transferred, not whether it sounds like")
    print("the show.")


if __name__ == "__main__":
    main()