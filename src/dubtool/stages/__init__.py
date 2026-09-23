"""Pipeline stages.

extract/align/mix/mux/diarize live here as plain functions/simple classes —
they're always ffmpeg/numpy/a one-speaker-stub, never swapped via config, so
they don't need the backend-registry treatment `interfaces.py` implementations
get. The model-backed stages (separate, transcribe, translate, synthesize)
live in `dubtool.backends` instead and are wired up by `dubtool.registry`.
"""
