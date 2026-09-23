"""Full pipeline, real models, real (synthetic) video. Slow — skipped by
default; run explicitly with `pytest -m integration`.

Manual QA note for whoever reviews the dubbed output this test produces:
listen for (1) the background 110Hz tone from make_sample_video.py still
being audible under the dubbed speech, not stripped out by Demucs, (2) the
dubbed Uzbek speech roughly matching the ~8s pacing of the three original
sentences rather than being wildly sped up/slowed down, and (3) the cloned
voice's timbre resembling assets/default_reference.wav rather than sounding
like a generic/flat voice.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dubtool.cli import main

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_VIDEO = FIXTURES_DIR / "sample_video.mp4"


@pytest.mark.integration
def test_full_pipeline_produces_output(tmp_path):
    if not SAMPLE_VIDEO.exists():
        pytest.skip(f"{SAMPLE_VIDEO} missing — run tests/integration/make_sample_video.py first")

    output = tmp_path / "dubbed.mp4"
    exit_code = main([str(SAMPLE_VIDEO), "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    assert output.stat().st_size > 0
