# UzAniDub

UzAniDub translates dialogue into Uzbek and synthesizes dubbed audio using a
short reference from each speaker. The Python pipeline can process video or
audio. The website lets you test a line, a voice sample, or a short scene, review
each generated line, and publish finished videos to a private library.

## Run locally

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

For multiple speakers, set `HF_TOKEN` after accepting the diarization model
terms. Without it, the pipeline treats the clip as one speaker. The workbench
shows which local components are available; missing models produce an error
rather than a synthetic example.

In one PowerShell terminal, from the repository root:

```powershell
.\.venv\Scripts\python.exe run_engine.py
```

This checkout has a local `.venv` prepared for Windows. For a fresh Python
environment, install the dependencies in `pyproject.toml` first. On Windows,
`resemblyzer`'s `webrtcvad` dependency may require C++ build tools; this
checkout uses the [prebuilt WebRTC VAD wheel](https://pypi.org/project/webrtcvad-wheels/)
and installs `resemblyzer` without its dependencies. It also uses `pyworld`
0.3.5 on Windows to avoid a NumPy binary mismatch. Use the same environment
for the CLI and engine.

In a second terminal:

```powershell
cd site
npm ci
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0000_cute_greymalkin.sql
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0001_pretty_shape.sql
npm run dev
```

To test cover and episode publishing locally, set
`$env:DUBTOOL_LOCAL_PUBLISHER = "local@localhost"` before `npm run dev`.

Apply each migration only once to a given local database. Open
`http://localhost:3000/create`. Short test scenes are limited to 512 MB in the
web workbench; the CLI is better for full episodes:

```powershell
.\.venv\Scripts\dubtool.exe input.mp4 -o output/episode-01.mp4 --names "Naruto,Pain"
```

The web engine binds to `127.0.0.1:8765` and keeps test inputs and results in
ignored `output/web-jobs/`. It does not publish them. The private hosted site
can use this engine when opened on the same computer; the browser may ask for
local network access. For other devices, hosted dubbing needs a separate,
authenticated model server because Cloudflare Workers cannot run these Python
models. Publishing finished videos is handled by the site.

See [the site guide](site/README.md), [voice model guide](docs/own-voice-model.md),
and [model licenses](THIRD_PARTY_LICENSES.md).
