# UzAniDub

A CLI tool that dubs a video from any source language into Uzbek: it
transcribes the dialogue, separates vocals from background audio/music,
translates each line, synthesizes Uzbek speech, time-aligns it back onto
the original timing, and mixes it back over the preserved background —
muxing the result into the source video (or writing plain audio if the
input has no video track).

```
video (any language)
  -> vocal/background separation (Demucs)
  -> speaker diarization (pyannote if its gated models are reachable,
     else an ungated Resemblyzer-based diarizer)
  -> transcription (faster-whisper)
  -> English slang/idiom normalization (for English-source segments)
  -> translation to Uzbek (MADLAD-400)
  -> Uzbek speech synthesis (CosyVoice2 + navoiy-tts)
  -> timing: every line spoken in full, sped up only via the TTS
  -> loudness that follows the actor, plus vocal effort on shouted lines
  -> stereo mix over the original music/effects + mux into the video
```

Every stage above is a swappable backend behind `dubtool.interfaces` (see
that file) — nothing in `pipeline.py` is tied to one specific model.

## Status

Functional end to end on real test files (business talk, anime, a short
documentary), with known rough edges: MADLAD occasionally hallucinates on
a specific input line rather than mistranslating it, and per-segment
timing correction sometimes has to trade off accepting timing drift
against distorting the voice further (see `config.py` for the full
reasoning). Voice cloning per character is **on by default**: every line
a character speaks clones from one reference of their voice, taken from
the source audio and kept in a persistent voice bank so they sound the
same across episodes (see `--no-clone-voices` to disable). It's combined
with emotion classification so nearly every line gets an explicit
delivery style, and with loudness that follows the original performance
— see [Emotion](#emotion).

Lines are never cut short to fit their slot. Uzbek usually runs longer
than the English it replaces, so a line may use the pause after it, is
sped up (through the TTS itself, never by stretching the audio) only as
far as it needs, and if it still runs long the next line waits. With
MADLAD that can leave a line starting up to a second or so late in dense
dialogue; the opt-in Gemini translator can be asked for lines that fit.

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

### Optional: pyannote diarization

Multi-speaker diarization works out of the box with an ungated diarizer
(`backends/diarize_resemblyzer.py`). pyannote gives better turn
boundaries and handles overlapping speech, but its models are gated: you
need a free Hugging Face account, a token, and access to **all** of
[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1),
[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and
[speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
(pyannote.audio 4.x needs the last one; unlike the other two, its access
is approved by hand, so it can take a while).

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

With a token, `diarize: auto` tries pyannote and falls back to the
ungated diarizer if access is refused. To skip the attempt, set
`backends: {diarize: resemblyzer}` in a config file.

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
| `--voice-bank-dir PATH` | Directory for the persistent cross-run voice bank, only relevant with voice cloning enabled (default: `./voice_bank`). Use one per show, e.g. `voice_bank/jjk` — matching is by voice similarity alone, so a shared bank can give one show's character another show's voice |
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

Every line gets a delivery style for CosyVoice2's `inference_instruct2`
(navoiy-tts's trained presets: calm, happy, sad, angry, surprised,
whispers, ...), chosen from how the actor delivered it
(`stages/emotion.py`): loudness measured against the whole scene rather
than each speaker's own baseline, so a character who shouts throughout
isn't read as calm; and the transcript's `!`/`?`, so an exclamation gets
an energetic reading and a short question reads as surprise instead of a
whisper.

The presets move pitch (+4-6 semitones for happy/angry/surprised over
calm, measured) but not much else, so two more things come from the
original performance (`stages/loudness.py`):

- **Loudness follows the actor.** Each line is levelled to the loudness
  the actor gave it, so a shout lands as a shout and the voice sits
  against the music where the original dialogue did. Levelling every line
  to a fixed target had squeezed a 9.5dB range into 2.3dB.
- **Vocal effort on shouted lines.** A scream is not only louder: effort
  pushes energy into the 2-4kHz presence range, which no instruction made
  the TTS do. Lines the actor delivered well above the scene get a
  presence boost in proportion, up to +6dB for a full scream. Each line's
  intensity and effort are logged.

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

With Gemini, names that spaCy detects automatically are passed to the model
as hints rather than kept verbatim (spaCy also tags show terms such as
"Limitless Technique" as names, which used to leave English in the dub).
The prompt asks for Uzbek in Latin script with no English left in, and a
line that comes back in Cyrillic or with source words in it is retried once.

## Running on Kaggle

`notebooks/kaggle_dub.ipynb` runs the whole pipeline on a free Kaggle GPU:
import it into a new Kaggle notebook, turn on **GPU T4** and **Internet**,
attach your videos as a private dataset, add `GEMINI_API_KEY` under
**Add-ons → Secrets** (or set `TRANSLATOR = "madlad"`), and **Run All**.
Dubbed files and logs land in `/kaggle/working/output`.

## Config file

Anything on `DubConfig` (see `src/dubtool/config.py`) can be set via a
YAML file passed with `--config`, instead of only via CLI flags — useful
for settings you want to reuse across runs of the same show (backend
choices, timing bounds, a persistent glossary, etc.).

## Website and local engine

Use Python 3.11 with the dependencies in `pyproject.toml`, Node 22+, and FFmpeg.
The model files are intentionally not committed. The paths expected by the
default configuration are in `src/dubtool/config.py`:

- `poc/CosyVoice` — [CosyVoice source](https://github.com/QwenAudio/CosyVoice)
- `poc/models/CosyVoice2-0.5B` — [base voice model](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)
- `poc/models/navoiy-tts/emotion_600h_joint.pt` — [Uzbek checkpoint](https://huggingface.co/aisha-org/navoiy-tts), plus `emotions_40h.json` and `uztts/normalize.py` from that repository

The [MADLAD translation model](https://huggingface.co/zenoverflow/madlad400-3b-mt-int8-float32)
downloads to `poc/models/madlad400-3b-mt-int8-float32` on its first use. The three model
repositories total roughly 10 GB before caches and Python packages, so check
free disk space before downloading. The source checkout must be compatible
with CosyVoice2; newer main-branch changes may need adaptation.

Multiple speakers work without a token, using the ungated diarizer; set
`HF_TOKEN` after accepting the pyannote terms for better turn boundaries (see
[Optional: pyannote diarization](#optional-pyannote-diarization)).

This checkout has a local `.venv` prepared for Windows. For a fresh Python
environment, install the dependencies in `pyproject.toml` first. On Windows,
`resemblyzer`'s `webrtcvad` dependency may require C++ build tools; this
checkout uses the [prebuilt WebRTC VAD wheel](https://pypi.org/project/webrtcvad-wheels/)
and installs `resemblyzer` without its dependencies. It also uses `pyworld`
0.3.5 on Windows to avoid a NumPy binary mismatch. Use the same environment
for the CLI.

To run the website:

```powershell
cd site
npm ci
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0000_cute_greymalkin.sql
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0001_pretty_shape.sql
npm run dev
```

Apply each migration only once to a given local database, then open
`http://localhost:3000` (the dev server picks the next free port if 3000 is
taken; the terminal shows the actual URL). Add anime and episodes at
`/upload`.

Dub an episode with the CLI, then publish the result through the upload page:

```powershell
.\.venv\Scripts\dubtool.exe input.mp4 -o output/episode-01.mp4 --names "Naruto,Pain"
```

`run_engine.py` starts a separate loopback HTTP service on `127.0.0.1:8765`
for running the dubbing models (short scenes up to 512 MB, results kept in
ignored `output/web-jobs/`). The website does not use it, and Cloudflare
Workers cannot run these Python models, so dubbing stays local.

See [the site guide](site/README.md), [voice model guide](docs/own-voice-model.md),
and [model licenses](THIRD_PARTY_LICENSES.md).

## License

dubtool itself is Apache-2.0 (see `LICENSE`). It depends on several
pretrained models and libraries with their own licenses — see
`THIRD_PARTY_LICENSES.md` before redistributing or deploying commercially,
especially if you swap in a non-default backend.
