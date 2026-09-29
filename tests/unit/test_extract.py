"""has_video_stream() must not mistake embedded cover art for real video."""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from dubtool.stages.extract import has_video_stream


def _probe(stdout: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class HasVideoStreamTests(unittest.TestCase):
    def test_audio_without_video_streams(self) -> None:
        with patch("subprocess.run", return_value=_probe("")):
            self.assertFalse(has_video_stream(Path("clip.mp3")))

    def test_cover_art_is_not_video(self) -> None:
        with patch("subprocess.run", return_value=_probe("1,1\n")):
            self.assertFalse(has_video_stream(Path("song.mp3")))

    def test_real_video_stream(self) -> None:
        with patch("subprocess.run", return_value=_probe("0,0\n")):
            self.assertTrue(has_video_stream(Path("scene.mp4")))

    def test_video_with_cover_art_is_still_video(self) -> None:
        with patch("subprocess.run", return_value=_probe("0,0\n2,1\n")):
            self.assertTrue(has_video_stream(Path("episode.mkv")))


if __name__ == "__main__":
    unittest.main()
