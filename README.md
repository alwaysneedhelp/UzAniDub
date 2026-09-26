# dubtool

A CLI tool that dubs a video from any source language into Uzbek: it
transcribes the dialogue, separates vocals from background audio/music,
translates each line, synthesizes Uzbek speech, time-aligns it back onto
the original timing, and mixes it back over the preserved background —
muxing the result into the source video (or writing plain audio if the
input has no video track).

```
video (any language)
  -> vocal/background separation (Demucs)
  -> speaker diarization (pyannote, optional)
  -> transcription (faster-whisper)
  -> English slang/idiom normalization (for English-source segments)
  -> translation to Uzbek (MADLAD-400)
  -> Uzbek speech synthesis (CosyVoice2 + navoiy-tts)
  -> time-alignment + loudness normalization
  -> mix + mux back into the video
```

Every stage above is a swappable backend behind `dubtool.interfaces` (see
that file) — nothing in `pipeline.py` is tied to one specific model.

## Status

Functional end to end on real test files (business talk, anime, a short
documentary), with known rough edges: MADLAD occasionally hallucinates on
a specific input line rather than mistranslating it, and per-segment
timing correction sometimes has to trade off accepting timing drift
against distorting the voice further (see `config.py` for the full
reasoning). Voice cloning per speaker is **on by default** (each segment
clones its speaker's own voice from the source audio, see `--no-clone-voices`
to disable); it's combined with acoustic emotion classification
(`stages/emotion.py`) so nearly every segment also gets an explicit
delivery style instead of a flat, generic reading — see
[Emotion](#emotion) below for why that combination, specifically, turned
out to matter.

## Install

Requirements: Python 3.11, `ffmpeg` on your `PATH` (`brew install ffmpeg`
/ `apt install ffmpeg`). A GPU is not required — CPU works, just slower.

```bash
git clone https://github.com/alwaysneedhelp/UzAniDub.git
cd UzAniDub
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Model weights (CosyVoice2, navoiy-tts, MADLAD-400, Demucs, Whisper, etc.)
download automatically on first use — expect a multi-GB download and
some time the first run.

### Optional: multi-speaker diarization

Real multi-speaker diarization (pyannote) requires a free Hugging Face
account, accepting the license terms on the
[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
model pages, and an access token:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Without a token, dubtool falls back to treating the whole clip as one
speaker — everything still works, it just won't distinguish speakers for
per-speaker features like voice cloning (every line clones from the same
one detected "speaker" instead of per-character).

## Usage

```bash
dubtool path/to/video.mp4
```

Writes the dubbed video to `output/<input filename>` by default.

### Useful flags

| Flag | What it does |
|---|---|
| `--target uz` | Target language code (default: `uz`) |
| `-o, --output PATH` | Output path (default: `output/<input filename>`) |
| `--config PATH` | YAML config file overriding any `DubConfig` default |
| `--num-speakers N` | Hint the diarizer with a known speaker count |
| `--names NAMES` | Character/place names to bias transcription toward and keep verbatim through translation. Either `"Naruto,Sasuke,Sakura"` or a path to a file, one name per line (see `assets/example_character_names.txt`) |
| `--glossary PATH` | YAML file of `{source term: fixed Uzbek translation}` pairs, enforced consistently on every occurrence — see [Glossary](#glossary) below |
| `--no-clone-voices` | Use the bundled generic voice for every line instead of cloning each speaker's own voice (cloning is on by default — see [Status](#status)) |
| `--voice-bank-dir PATH` | Directory for the persistent cross-run voice bank, only relevant with voice cloning enabled (default: `./voice_bank`) |
| `--no-voice-bank` | Disable the persistent voice bank for this run |
| `--emotion NAME` | Force one TTS delivery style for every segment instead of each segment's own acoustically-detected style (see [Emotion](#emotion)) |
| `--keep-intermediate` | Keep extracted audio, separated tracks, etc. for debugging |
| `-v, --verbose` | Debug-level logging |

Run `dubtool --help` for the exact current list.

## Glossary

A general-purpose translation model has no memory across calls — it can
translate the same recurring term two different ways in two different
segments of the same video, and it may mangle a name it's never seen
before (verified on real content). The glossary fixes both problems by
forcing a pre-decided, consistent Uzbek rendering for every entry you give
it:

```yaml
# a name that should pass through unchanged is an entry mapping to itself
Gojo: Gojo
# a term that should be translated maps to its fixed rendering
Cursed Energy: La'nat energiyasi
```

Every occurrence of a glossary entry is protected before translation and
restored to its fixed rendering afterward, so the same term always comes
out the same way. `assets/jjk_glossary.yaml` is a ready-made example built
for *Jujutsu Kaisen* (character names + show terminology, researched since
no official Uzbek dub exists to draw from):

```bash
dubtool episode1.mp4 --glossary assets/jjk_glossary.yaml
```

`--names` is a shortcut for glossary entries that should just pass
through unchanged (no fixed translation needed) — both flags write into
the same underlying glossary and can be combined.

## Emotion

Each segment's own original audio is scored for pitch/energy relative to
its speaker's own baseline elsewhere in the file (`stages/emotion.py`),
and routed into CosyVoice2's `inference_instruct2` with a matching named
delivery style (calm, happy, sad, angry, surprised, whispers, nervous,
tired) — the
cloned reference clip (see above) still supplies voice *identity* in that
same call, so this isn't an either/or against cloning.

Earlier, only acoustically extreme segments got an explicit style; a
near-baseline segment fell back to `inference_zero_shot` (implicit,
"natural" prosody transfer from the reference clip) instead. That
produced consistently flat/robotic output even with a real, expressive
per-segment reference clip — the navoiy-tts checkpoint was fine-tuned
mainly on discrete named emotion presets, not on zero-shot delivery
transfer, so zero-shot was the actual weak link, not voice cloning
itself. Now nearly every segment gets classified (near-baseline delivery
maps to `calm` rather than nothing), so instruct2 handles almost
everything; zero-shot is only reached when a segment has no usable pitch
signal at all (near-silent/unvoiced).

## Slang/idiom normalization

MADLAD-400 (like most classic MT models) is trained mostly on formal
text and mistranslates casual English slang it rarely saw in training —
"for real" is a real example that came out wrong. Rather than swap the
whole translation backend for a much heavier LLM (which on this CPU-only
pipeline would add real per-segment latency, and isn't a guaranteed win
for Uzbek specifically — see `stages/idioms.py`), English-source segments
get a cheap normalization pass first that rewrites known slang/idioms
into plainer English before translation. It's intentionally a small,
non-exhaustive list — extend `stages/idioms.py` as new mistranslations
turn up.

For a stronger fix than word-swapping, `backends.translate: gemini` in a
`--config` file switches translation to the Gemini API instead of
MADLAD-400 — an instruction-following model can be asked directly for a
natural, idiomatic rendering (slang included) instead of a literal one.
It's opt-in, not the default: unlike every default backend, it's a paid
cloud call that needs its own API key and sends the source dialogue to
Google (see `THIRD_PARTY_LICENSES.md`). Get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and set
it in `.env`:

```
GEMINI_API_KEY=...
```

```yaml
# config.yaml
backends:
  translate: gemini
```

```bash
dubtool episode1.mp4 --config config.yaml
```

## Config file

Anything on `DubConfig` (see `src/dubtool/config.py`) can be set via a
YAML file passed with `--config`, instead of only via CLI flags — useful
for settings you want to reuse across runs of the same show (backend
choices, timing bounds, a persistent glossary, etc.).

## License

dubtool itself is Apache-2.0 (see `LICENSE`). It depends on several
pretrained models and libraries with their own licenses — see
`THIRD_PARTY_LICENSES.md` before redistributing or deploying commercially,
especially if you swap in a non-default backend.
