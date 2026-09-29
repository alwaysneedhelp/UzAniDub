"""HTTP contract checks that do not load or download speech models."""
from __future__ import annotations

import json
import base64
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from dubtool import web_service
from dubtool.web_service import Handler


class WebServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health_and_allowed_origin(self) -> None:
        request = Request(self.base + "/health", headers={"Origin": "http://localhost:3000"})
        with urlopen(request) as response:
            health = json.load(response)
            self.assertTrue(health["online"])
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:3000")

    def test_other_browser_origins_are_rejected(self) -> None:
        request = Request(self.base + "/health", headers={"Origin": "https://untrusted.example"})
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 403)

    def test_private_site_can_request_local_engine_access(self) -> None:
        origin = "https://uzanidub-series.jasur0407.chatgpt.site"
        request = Request(self.base + "/health", method="OPTIONS", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        })
        with urlopen(request) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
            self.assertEqual(response.headers["Access-Control-Allow-Private-Network"], "true")

    def test_invalid_upload_rejected_before_job_is_queued(self) -> None:
        request = Request(self.base + "/jobs/dub", data=b"test", method="POST", headers={
            "Origin": "http://localhost:3000",
            "X-Dub-Filename": "unsafe.exe",
            "X-Dub-Options-B64": base64.b64encode(json.dumps({"names": ["Oʻzbekiston"]}).encode("utf-8")).decode(),
        })
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 400)
        self.assertIn("Choose a video (MP4", context.exception.read().decode())

    def test_dub_rejected_before_upload_when_models_are_missing(self) -> None:
        request = Request(self.base + "/jobs/dub", data=b"sample", method="POST", headers={
            "Origin": "http://localhost:3000", "X-Dub-Filename": "clip.mp3",
        })
        with patch.object(web_service, "readiness", return_value={
            "readyForDub": False,
            "checks": {"ffmpeg": True, "Demucs separation API": False,
                       "CosyVoice model": False, "Multi-speaker token": False},
        }):
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
        self.assertEqual(context.exception.code, 503)
        message = json.load(context.exception)["error"]
        self.assertIn("Demucs separation API", message)
        self.assertIn("CosyVoice model", message)
        self.assertNotIn("Multi-speaker token", message)

    def test_transcript_source_audio_can_be_previewed(self) -> None:
        job_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        with tempfile.TemporaryDirectory() as temporary, patch.object(web_service, "JOBS_DIR", Path(temporary)):
            (Path(temporary) / job_id).mkdir()
            (Path(temporary) / job_id / "input.mp3").write_bytes(b"sample")
            web_service.save({"id": job_id, "input": "input.mp3", "segments": [{"original": "Hello"}]})
            request = Request(self.base + f"/jobs/{job_id}/media/input.mp3", headers={
                "Origin": "http://localhost:3000", "Range": "bytes=0-2",
            })
            with urlopen(request) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Type"], "audio/mpeg")
                self.assertEqual(response.read(), b"sam")


if __name__ == "__main__":
    unittest.main()
