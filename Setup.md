
# Setup for voice_clone_test.py

This is a one-off test, not the full pipeline. Expect ~20-30 minutes of
setup the first time, mostly checkpoint downloads.

## 1. System dependencies

- Python 3.9-3.11 recommended (OpenVoice and Demucs can be picky about newer
  Python versions).
- `ffmpeg` must be on your PATH.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: install from ffmpeg.org and add to PATH.
- A GPU is not required. CPU works, just slower (Demucs and the tone
  conversion step will each take longer).

## 2. Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. OpenVoice (not on PyPI in a stable pinned form -- clone the repo)

This is what `voice_clone_test.py` was missing when you saw
`No module named 'openvoice'` -- the package has to be installed
separately, it doesn't come from `requirements.txt`.

**Easiest path: run the automated script.** With the same venv from step 2
still active:

```bash
chmod +x setup_openvoice.sh
./setup_openvoice.sh
```

This clones the public OpenVoice repo over HTTPS (no GitHub login needed,
ever -- it's a public repo), installs the `openvoice` package and its
MeloTTS dependency into your active venv, and downloads the verified V2
checkpoint archive automatically. Takes a while the first time (the
checkpoint alone is a few hundred MB).

**Manual path**, if you'd rather see each step or the script fails on your
machine:

```bash
git clone https://github.com/myshell-ai/OpenVoice.git
cd OpenVoice
pip install -e .
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
curl -L -o checkpoints_v2.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip checkpoints_v2.zip
```

You should end up with a `checkpoints_v2/converter/` folder containing
`config.json` and `checkpoint.pth`.

Two ways to make voice_clone_test.py find OpenVoice:

- **Easiest:** copy `voice_clone_test.py` into the root of the cloned
  `OpenVoice` folder (next to its own `openvoice/` package) and run it
  from there, with `checkpoints_v2/converter/` also in that same folder.
- **Or:** leave the script wherever you like and set two environment
  variables before running:
  ```bash
  export OPENVOICE_REPO=/path/to/OpenVoice
  export OPENVOICE_CKPT_V2=/path/to/OpenVoice/checkpoints_v2/converter
  ```

## 4. Find a clean reference clip

Open "Naruto scene.mp4" in any video player and find a short (5-15 second)
stretch where:

- Only one character is talking (no overlapping dialogue)
- There's minimal music/sound effects under the line
- The delivery is relatively normal speech, not full-volume shouting
  (shouting is a harder case -- worth a second test once the calm case
  works, but don't start there)

Note the start and end time in seconds.

## 5. Run it

```bash
python voice_clone_test.py \
  --video "Naruto scene.mp4" \
  --ref-start 12.0 --ref-end 18.5 \
  --text "Salom, bugun ajoyib kun, sen buni uddalaysan."
```

First run will download the MMS Uzbek TTS model (~150MB) automatically.

If Demucs separation is slow or you just want a fast first pass, add
`--skip-separation` to slice the reference directly from the raw audio
(background sound will leak into the reference, which will likely hurt
cloning quality -- but it's a fast way to sanity-check the rest of the
pipeline before waiting on Demucs).

## 6. What you'll get

Three files under `work/`:

1. `reference.wav` -- the original character's voice, isolated
2. `uzbek_generic.wav` -- correct Uzbek speech, generic TTS voice
3. `uzbek_cloned.wav` -- correct Uzbek speech, attempted cloned voice

Listen to #3 against #1. You're checking whether the *voice quality/timbre*
transferred -- not whether it sounds exactly like the show, and not
against your memory of the dub. Some things to listen for specifically:

- Does it sound like a plausible variant of the reference voice, or does
  it sound like the generic voice with a slight "flavor" added (weak
  transfer)?
- Are there robotic artifacts, warbling, or metallic distortion (common
  failure mode when the reference clip is noisy or too short)?
- Is the Uzbek still intelligible, or did the conversion step mangle
  pronunciation (would suggest the phoneme mismatch problem is real)?

## If it sounds bad

A few things to try before concluding the approach doesn't work:

- Use a longer, cleaner reference clip (15-20s of uninterrupted speech,
  no music).
- Try `--skip-separation` off vs on -- sometimes Demucs artifacts hurt
  more than raw bleed-through, sometimes the reverse.
- Try OpenVoice V1 instead of V2 (different checkpoint, sometimes handles
  certain voices differently) -- same repo, different checkpoint folder.
- If it's consistently bad specifically on Uzbek phonemes (o', g', q, h
  sounds coming out wrong or muffled), that's evidence the cross-lingual
  claim doesn't hold up well for Uzbek specifically, and it's worth
  reporting back so we can look at fine-tuning options instead.
