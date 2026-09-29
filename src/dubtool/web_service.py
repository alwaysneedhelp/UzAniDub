"""Loopback HTTP bridge for testing the real dubbing models from the site.

Run with ``PYTHONPATH=src python -m dubtool.web_service``. The browser UI is
the client; this process owns model loading and media files. It deliberately
binds to loopback, and never exposes a public unauthenticated model endpoint.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "output" / "web-jobs"
MAX_UPLOAD = 512 * 1024 * 1024  # Short scenes in the browser; use the CLI for episodes.
ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://uzanidub-series.jasur0407.chatgpt.site",
}
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
               ".flac": "audio/flac", ".ogg": "audio/ogg", ".opus": "audio/ogg"}
VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
ALLOWED_ORIGINS.update(filter(None, os.environ.get("DUBTOOL_ALLOWED_ORIGINS", "").split(",")))
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dubtool-web")
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
STAGES = {
    1: "Extracting audio", 2: "Separating dialogue", 3: "Finding speakers",
    4: "Sampling voices", 5: "Transcribing", 6: "Translating",
    7: "Generating Uzbek voices", 8: "Mixing final audio",
}


def readiness() -> dict:
    from dubtool.config import DubConfig
    from dubtool.backends.translate_madlad import local_model_ready

    model = DubConfig().models
    def cached(repo: str) -> bool:
        # An interrupted download leaves a snapshot folder holding only the
        # small config files, so require the weights file itself.
        try:
            from huggingface_hub import snapshot_download
            return (Path(snapshot_download(repo, local_files_only=True)) / "model.bin").is_file()
        except Exception:
            return False

    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "Translation packages": all(importlib.util.find_spec(name) is not None for name in
                                    ("ctranslate2", "transformers", "sentencepiece", "huggingface_hub")),
        "Translation model cached": local_model_ready() or cached(model.madlad_ct2_model),
        "Speech recognition model cached": cached(f"Systran/faster-whisper-{model.whisper_model}"),
        "Voice runtime packages": all(importlib.util.find_spec(name) is not None for name in
                                      ("numpy", "torch", "torchaudio", "soundfile", "conformer", "diffusers",
                                       "lightning", "hyperpyyaml", "onnxruntime", "inflect", "x_transformers",
                                       "wetext", "pyworld", "omegaconf", "hydra",
                                       # Imported by CosyVoice's bundled Matcha-TTS at model load.
                                       "gdown", "wget", "rootutils", "pyarrow")),
        "Audio pipeline packages": all(importlib.util.find_spec(name) is not None for name in
                                       ("librosa", "demucs", "scipy", "spacy", "resemblyzer", "faster_whisper")),
        "Demucs separation API": (importlib.util.find_spec("demucs") is not None and
                                  importlib.util.find_spec("demucs.api") is not None and
                                  importlib.util.find_spec("sphn") is not None),
        "CosyVoice source": (ROOT / model.cosyvoice_source_dir).is_dir(),
        "CosyVoice model": all((ROOT / model.cosyvoice_base_dir / name).is_file() for name in
                                ("cosyvoice2.yaml", "flow.pt", "hift.pt", "llm.pt")),
        "Uzbek voice checkpoint": all(path.is_file() for path in
            (ROOT / model.navoiy_checkpoint,
             ROOT / model.navoiy_checkpoint.parent / "emotions_40h.json",
             ROOT / model.navoiy_checkpoint.parent / "uztts" / "normalize.py")),
        "Multi-speaker token": bool(os.environ.get("HF_TOKEN")),
    }
    return {"online": True, "checks": checks,
            "readyForTranslation": checks["Translation packages"] and checks["Translation model cached"],
            "readyForTranscription": checks["Speech recognition model cached"] and importlib.util.find_spec("faster_whisper") is not None,
            "readyForVoice": all(checks[k] for k in
            ("Voice runtime packages", "CosyVoice source", "CosyVoice model", "Uzbek voice checkpoint")),
            "readyForDub": all(checks[k] for k in checks if k != "Multi-speaker token")}


def save(job: dict) -> None:
    job_dir = JOBS_DIR / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    with JOBS_LOCK:
        JOBS[job["id"]] = dict(job)
        pending = job_dir / "job.json.tmp"
        pending.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        pending.replace(job_dir / "job.json")


def load(job_id: str) -> dict | None:
    with JOBS_LOCK:
        if job_id in JOBS:
            return dict(JOBS[job_id])
    path = JOBS_DIR / job_id / "job.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_job(job_id: str) -> None:
    job = load(job_id)
    if job is None:
        return
    folder = JOBS_DIR / job_id
    job.update(status="running", stage="Loading models", error=None)
    save(job)
    try:
        from dubtool.config import DubConfig

        config = DubConfig()
        config.source_language = job.get("sourceLanguage") or None
        config.num_speakers = job.get("speakers") or None
        config.proper_nouns = job.get("names", [])
        config.tts_emotion = job.get("emotion") or None
        if not job.get("voiceBank", True):
            config.voice_bank_dir = None

        if job["kind"] == "translate":
            from dubtool.backends.translate_madlad import MadladTranslator

            job.update(stage="Translating", progress=35)
            save(job)
            translated = MadladTranslator(model_name=config.models.madlad_ct2_model).translate(
                job["text"], source_lang=config.source_language or "auto", target_lang="uz")
            job["translation"] = translated
        elif job["kind"] == "transcribe":
            from dubtool.backends.transcribe_whisper import FasterWhisperTranscriber
            from huggingface_hub import snapshot_download

            job.update(stage="Transcribing", progress=35)
            save(job)
            model_path = config.models.whisper_model
            try:
                model_path = snapshot_download(f"Systran/faster-whisper-{model_path}", local_files_only=True)
            except Exception:
                pass  # FasterWhisperTranscriber can download a missing model normally.
            transcriber = FasterWhisperTranscriber(model_size=model_path,
                                                   language=config.source_language,
                                                   proper_nouns=config.proper_nouns)
            segments = transcriber.transcribe(folder / job["input"], [])
            job["segments"] = [{"speaker": segment.speaker_id, "start": round(segment.start, 2),
                                "end": round(segment.end, 2), "original": segment.text,
                                "uzbek": ""} for segment in segments]
            job["detectedLanguage"] = segments[0].detected_language if segments else None
        elif job["kind"] == "voice":
            import soundfile as sf
            from dubtool.backends.tts_cosyvoice import CosyVoiceNavoiyTTS

            job.update(stage="Generating Uzbek voice", progress=35)
            save(job)
            tts = CosyVoiceNavoiyTTS(
                cosyvoice_source_dir=config.models.cosyvoice_source_dir,
                base_model_dir=config.models.cosyvoice_base_dir,
                checkpoint_path=config.models.navoiy_checkpoint,
            )
            audio = tts.synthesize(job["text"], folder / job["input"],
                                   reference_text=job["referenceText"], emotion=config.tts_emotion)
            sf.write(folder / "voice.wav", audio, tts.sample_rate)
            job["output"] = "voice.wav"
        else:
            import soundfile as sf
            from dubtool.pipeline import run
            from dubtool.registry import build_backends

            class StageHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    if record.name != "dubtool.pipeline":
                        return
                    match = re.search(r"Stage (\d)/8:", record.getMessage())
                    if match:
                        step = int(match.group(1))
                        job.update(stage=STAGES[step], progress=5 + step * 10)
                        save(job)

            handler = StageHandler()
            logger = logging.getLogger("dubtool.pipeline")
            logger.addHandler(handler)
            try:
                config.keep_intermediate = True
                config.work_dir = folder / "work"
                result = run(folder / job["input"], folder / "dubbed.mp4", config, build_backends(config))
            finally:
                logger.removeHandler(handler)
            job["output"] = result.output_video.name
            lines = []
            for index, segment in enumerate(result.segments):
                audio_name = f"line-{index + 1:03d}.wav"
                if segment.synthesized_audio is not None and segment.synthesized_sr:
                    sf.write(folder / audio_name, segment.synthesized_audio, segment.synthesized_sr)
                lines.append({"speaker": segment.speaker_id, "start": round(segment.start, 2),
                              "end": round(segment.end, 2), "original": segment.text,
                              "uzbek": segment.translated_text, "audio": audio_name})
            job["segments"] = lines
        job.update(status="done", stage="Ready to review", progress=100)
    except Exception as error:
        logging.exception("Web dubbing job %s failed", job_id)
        job.update(status="failed", stage="Stopped", error=str(error), detail=traceback.format_exc(limit=2))
    save(job)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        logging.info("web engine: " + format, *args)

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def _headers(self, code: int, content_type: str, size: int | None = None) -> None:
        self.send_response(code)
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Type", content_type)
        if size is not None:
            self.send_header("Content-Length", str(size))
        self.end_headers()

    def _json(self, data: dict | list, code: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._headers(code, "application/json; charset=utf-8", len(raw))
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        if not self._origin_ok():
            self._json({"error": "Origin not allowed"}, 403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", "http://localhost:3000"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Dub-Options-B64, X-Dub-Filename")
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._origin_ok():
            self._json({"error": "Origin not allowed"}, 403)
            return
        parts = urlsplit(self.path).path.strip("/").split("/")
        if parts == ["health"]:
            self._json(readiness())
            return
        if parts == ["jobs"]:
            jobs = [json.loads(p.read_text(encoding="utf-8")) for p in JOBS_DIR.glob("*/job.json")]
            self._json({"jobs": sorted(jobs, key=lambda item: item.get("createdAt", 0), reverse=True)[:30]})
            return
        if len(parts) >= 2 and parts[0] == "jobs" and re.fullmatch(r"[a-f0-9-]{36}", parts[1]):
            job = load(parts[1])
            if job is None:
                self._json({"error": "Job not found"}, 404)
                return
            if len(parts) == 2:
                self._json(job)
                return
            if len(parts) == 4 and parts[2] == "media" and parts[3] in {job.get("output"), job.get("input"), *(line.get("audio") for line in job.get("segments", []))}:
                self._media(JOBS_DIR / parts[1] / parts[3])
                return
        self._json({"error": "Not found"}, 404)

    def _media(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": "Media not found"}, 404)
            return
        size = path.stat().st_size
        mime = {**VIDEO_TYPES, **AUDIO_TYPES}.get(path.suffix, "application/octet-stream")
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        start = int(match.group(1)) if match else 0
        end = min(int(match.group(2)), size - 1) if match and match.group(2) else size - 1
        if start >= size or end < start:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        count = end - start + 1
        self.send_response(206 if match else 200)
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(count))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if match:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            while count:
                chunk = source.read(min(count, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                count -= len(chunk)

    def _discard_body(self, length: int) -> None:
        # Answering before the browser has finished sending its upload makes
        # the socket close with unread data, which the browser reports as a
        # generic network failure instead of showing our error message.
        while length > 0:
            chunk = self.rfile.read(min(length, 1024 * 1024))
            if not chunk:
                break
            length -= len(chunk)

    def do_POST(self) -> None:
        if not self._origin_ok():
            self._json({"error": "Origin not allowed"}, 403)
            return
        kind = urlsplit(self.path).path.removeprefix("/jobs/")
        if kind not in {"dub", "voice", "translate", "transcribe"}:
            self._json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json({"error": "Invalid input size"}, 400)
            return
        if length < 1 or length > MAX_UPLOAD:
            self._json({"error": "Input must be between 1 byte and 512 MB"}, 413)
            return
        try:
            encoded = self.headers.get("X-Dub-Options-B64", "")
            if len(encoded) > 8192:
                raise ValueError("Options are too long")
            options = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8") if encoded else "{}")
            if not isinstance(options, dict):
                raise ValueError("Options must be an object")
            text = str(options.get("text", "")).strip()
            if kind in {"voice", "translate"} and not (1 <= len(text) <= 1000):
                raise ValueError("Enter 1–1000 characters of text")
            speakers = options.get("speakers")
            if speakers not in (None, "") and (not isinstance(speakers, int) or not 1 <= speakers <= 12):
                raise ValueError("Speaker count must be 1–12")
            names = options.get("names", [])
            if not isinstance(names, list) or len(names) > 100 or any(not isinstance(x, str) or len(x) > 80 for x in names):
                raise ValueError("Invalid character names")
            filename = Path(self.headers.get("X-Dub-Filename", "input.mp4")).name
            suffix = Path(filename).suffix.lower()
            if kind in {"dub", "transcribe"} and suffix not in VIDEO_TYPES.keys() | AUDIO_TYPES.keys():
                raise ValueError("Choose a video (MP4, WebM, MOV, MKV) or audio file (WAV, MP3, M4A, AAC, FLAC, OGG, Opus)")
            if kind == "voice" and suffix not in AUDIO_TYPES:
                raise ValueError("Choose a WAV, MP3, M4A, AAC, FLAC, OGG, or Opus voice reference")
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            self._discard_body(length)
            self._json({"error": str(error)}, 400)
            return
        if kind in {"dub", "voice"}:
            health = readiness()
            ready = health["readyForDub" if kind == "dub" else "readyForVoice"]
            if not ready:
                needed = (health["checks"] if kind == "dub" else
                          {key: health["checks"][key] for key in
                           ("Voice runtime packages", "CosyVoice source", "CosyVoice model", "Uzbek voice checkpoint")})
                missing = ", ".join(key for key, available in needed.items()
                                    if not available and key != "Multi-speaker token")
                self._discard_body(length)
                self._json({"error": f"Set up the missing engine requirements before starting: {missing}"}, 503)
                return
        job_id = str(uuid.uuid4())
        folder = JOBS_DIR / job_id
        folder.mkdir(parents=True, exist_ok=True)
        input_name = f"input{suffix}" if kind != "translate" else None
        if input_name:
            with (folder / input_name).open("wb") as target:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        self._json({"error": "Incomplete upload"}, 400)
                        return
                    target.write(chunk)
                    remaining -= len(chunk)
        else:
            self.rfile.read(length)
        job = {"id": job_id, "kind": kind, "status": "queued", "stage": "Waiting for engine",
               "progress": 0, "createdAt": time.time(), "text": text, "input": input_name,
               "referenceText": str(options.get("referenceText", ""))[:3000],
               "sourceLanguage": str(options.get("sourceLanguage", ""))[:12],
               "speakers": speakers or None, "names": names,
               "emotion": str(options.get("emotion", ""))[:30],
               "voiceBank": options.get("voiceBank", True) is not False}
        save(job)
        EXECUTOR.submit(run_job, job_id)
        self._json(job, 202)


def main() -> None:
    from dotenv import load_dotenv

    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    ALLOWED_ORIGINS.update(filter(None, os.environ.get("DUBTOOL_ALLOWED_ORIGINS", "").split(",")))
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Dubtool engine: http://127.0.0.1:8765", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
