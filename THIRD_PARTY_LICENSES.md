# Third-party models and dependencies

dubtool itself is Apache-2.0 (see `LICENSE`). It depends on several
pretrained models and libraries with their own, separate licenses. This
file tracks each one so users can make an informed call about redistribution
and commercial use — several models in this space carry non-commercial
restrictions that are easy to miss.

## Models used by default (all permissive)

| Model | License | Used for | Notes |
|---|---|---|---|
| [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) (Alibaba FunAudioLLM) | Apache-2.0 | Base zero-shot voice-cloning TTS engine | Provides the flow-matching decoder, HiFiGAN vocoder, and CAMPPlus speaker-embedding model — the components that actually do voice cloning. |
| [aisha-org/navoiy-tts](https://huggingface.co/aisha-org/navoiy-tts) | Apache-2.0 | Uzbek fine-tune (LLM/text-to-token component only) | Swaps in Uzbek pronunciation on top of the untouched CosyVoice2 cloning machinery above. Ships its own stdlib-only Uzbek text normalizer (`uztts/normalize.py`). |
| [zenoverflow/madlad400-3b-mt-int8-float32](https://huggingface.co/zenoverflow/madlad400-3b-mt-int8-float32) (int8 CTranslate2 quantization of [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt)) | Apache-2.0 | Default translation backend, any source language → Uzbek in one hop (`<2uz>` target tag) | The vanilla fp32 checkpoint (also Apache-2.0) is 11.76GB — too large for this machine's disk budget. This quantization is ~3GB and reuses `ctranslate2`, already a dependency for faster-whisper, so no new heavy dependency. Chosen over the OPUS-MT pair below after real testing showed OPUS-MT dropping large chunks of meaning and occasionally hallucinating unrelated phrases. |
| [Demucs](https://github.com/facebookresearch/demucs) (htdemucs) | MIT | Vocal/background separation | |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [openai-whisper](https://github.com/openai/whisper) | MIT | Transcription with word-level timestamps | Whisper's own Uzbek recognition quality is weak (see PoC notes) — used here for source-language transcription, where it's typically much stronger. |
| [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) + [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) | MIT | Multi-speaker diarization (`backends.diarize: pyannote`, or `auto` when an `HF_TOKEN` is set) | Gated on Hugging Face — using it requires a free account, accepting the license terms on both model pages, and an access token (see README). The gating is for usage tracking/contact, not a licensing restriction: the pipeline and weights are genuinely MIT, free for commercial use. Falls back to the single-speaker stub when no token is configured, so this dependency is never hard-required. |
| [spaCy](https://github.com/explosion/spaCy) + [en_core_web_sm](https://huggingface.co/spacy/en_core_web_sm) | MIT | Automatic proper-noun (character/place name) detection from the transcript (`stages/auto_names.py`) | English-only NER model — auto-detection only fires for English source content; falls back to whatever's explicitly supplied via `--names` otherwise. A pretrained English text-*emotion* classifier was considered for the emotion-detection feature below and rejected specifically for this same English-only limitation, on top of having no clearly stated license. |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (pulls in `webrtcvad`, MIT) | Apache-2.0 | Cross-run voice identity matching for the persistent voice bank (`stages/voice_bank.py`) | Same embedding model already used informally during development to sanity-check cloning fidelity — promoted to a real dependency once the voice bank needed it for matching, not just one-off validation. |

## Other permissively-licensed backends available, but not default

| Model | License | Why not default |
|---|---|---|
| [Helsinki-NLP/opus-mt-en-trk](https://huggingface.co/Helsinki-NLP/opus-mt-en-trk) + [Helsinki-NLP/opus-mt-mul-en](https://huggingface.co/Helsinki-NLP/opus-mt-mul-en) | Apache-2.0 | Was the default translator before MADLAD-400's int8 quantization was found. Much lighter (~600MB combined vs ~3GB), but noticeably lossier — its own benchmark reports eng-uzb BLEU ~3.4 vs ~35 for eng-tur, and real testing showed it dropping large chunks of meaning and occasionally hallucinating unrelated phrases. Kept as `backends.translate: opus_mt` for disk-constrained setups. |

## Models flagged non-commercial — NOT used by default

| Model | License | Why it's excluded from defaults |
|---|---|---|
| [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | **CC-BY-NC-4.0** | Non-commercial only. Available as an opt-in translation backend (`backends.translate: nllb` in config) for users who accept that restriction, but must never be the default in a tool that wants to stay commercially redistributable. |
| [facebook/mms-tts-uzb-script_cyrillic](https://huggingface.co/facebook/mms-tts-uzb-script_cyrillic) | **CC-BY-NC-4.0** | Evaluated during the research phase as a possible Uzbek TTS base for a two-step voice-conversion approach. Not used at all — CosyVoice2 + navoiy-tts (both Apache-2.0) turned out to support voice cloning natively, making this unnecessary. Documented here only so nobody reaches for it later without noticing the license. |
| [uzlm/sayro-tts-1.7B](https://huggingface.co/uzlm/sayro-tts-1.7B) | Custom "sayro-terms-of-use" (restrictive, no redistribution clarity) | Ruled out during research: no reference-audio voice cloning support (fixed speaker IDs only) and an unclear, non-standard license. Not used anywhere. |

## Cloud APIs used only as opt-in backends

| Service | Terms | Used for | Why not default |
|---|---|---|---|
| [Gemini API](https://ai.google.dev/gemini-api/docs) (Google) | Proprietary, paid, governed by [Google's API Terms of Service](https://ai.google.dev/gemini-api/terms) — not an open-weight model, no redistributable license to track | Opt-in translation backend (`backends.translate: gemini`, see `backends/translate_gemini.py`) — better slang/idiom handling than MADLAD-400 (see `stages/idioms.py` for the cheaper default fix) | Every other default backend in this project is a local, permissively-licensed model with no per-run cost, no network dependency, and no third party seeing your content. Gemini is the opposite on all three counts: it costs money per call, requires internet access and a personal API key, and sends the source dialogue text to Google. Fine as an opt-in for someone who's made that tradeoff deliberately; wrong as a default for a tool whose whole design point elsewhere is avoiding exactly this kind of dependency. |

## Datasets referenced (not currently bundled, relevant for future fine-tuning)

| Dataset | License |
|---|---|
| [ISSAI Uzbek Speech Corpus](https://issai.nu.edu.kz/uzbek-asr/) | CC-BY-4.0 |
| [Mozilla Common Voice (Uzbek)](https://commonvoice.mozilla.org/) | CC0 |

## Reviewing before you redistribute or deploy commercially

If you swap in a different backend (via `config.backends`), check its
license before deploying — this table only covers what dubtool ships with
by default. In particular, do not swap the translation backend to
`nllb` in anything you intend to redistribute or run commercially without
separately clearing that with Meta, and treat `gemini` as sending your
source content to a third party (Google) — don't enable it for anything
sensitive or copyrighted without accounting for that.
