"""Orchestrates the dubbing pipeline. Talks only to `interfaces.py` shapes and
the plain-function stages (extract/align/mix/mux/reference aren't pluggable —
they're always ffmpeg/numpy/signal-processing heuristics, not swappable
models — so they're functions, not backend classes).

This is the step-3 (single-speaker, WITH cloning) state: diarize is still a
stub that returns one speaker for the whole clip, but that speaker's own
voice is now cloned from a reference clip extracted from their isolated
vocal track (see stages/reference.py), falling back to the bundled generic
default only if extraction fails for some reason. Step 4 swaps the stub
diarizer for pyannote so `reference_clips` has more than one entry — nothing
here needs to change for that, which is the point of keeping this behind
interfaces.py.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dubtool.config import DubConfig
from dubtool.interfaces import Diarizer, Separator, Transcriber, Translator, TTSBackend
from dubtool.stages import (
    align,
    auto_names,
    emotion,
    extract,
    mix,
    mux,
    proper_nouns,
    reference,
    resegment,
    voice_bank,
)
from dubtool.types import DubbingResult, Segment

log = logging.getLogger("dubtool.pipeline")


@dataclass
class Backends:
    separator: Separator
    diarizer: Diarizer
    transcriber: Transcriber
    translator: Translator
    tts: TTSBackend


class NoSpeechDetectedError(RuntimeError):
    pass


class UnreasonableSpeakerCountError(RuntimeError):
    pass


def run(
    video_path: Path,
    output_path: Path,
    config: DubConfig,
    backends: Backends,
) -> DubbingResult:
    # A fixed default work_dir used to make different invocations silently
    # clobber/mix each other's intermediate files (hit this for real while
    # debugging output quality). Allocate a fresh temp dir per run unless the
    # caller explicitly pinned one via config.
    work_dir = config.work_dir or Path(tempfile.mkdtemp(prefix="dubtool_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    if config.keep_intermediate:
        log.info("intermediate artifacts will be kept at: %s", work_dir)

    log.info("Stage 1/8: extracting audio from %s", video_path)
    full_audio = extract.extract_audio(video_path, work_dir / "full_audio.wav")
    has_video = extract.has_video_stream(video_path)
    if not has_video:
        log.info("input has no video stream (audio-only source) — will write dubbed audio directly instead of muxing")

    log.info("Stage 2/8: separating vocals from background")
    vocals_path, background_path = backends.separator.separate(full_audio, work_dir)

    log.info("Stage 3/8: diarizing")
    diarized_segments = backends.diarizer.diarize(vocals_path, num_speakers=config.num_speakers)
    _validate_speaker_count(diarized_segments, config.num_speakers)
    segments = diarized_segments

    log.info("Stage 4/8: extracting per-speaker reference clips for voice cloning")
    # Computed from the diarized (coarse) segments' speaker_ids, but *applied*
    # after transcribe()+resegment() below — both return fresh Segment lists
    # (re-split at ASR boundaries, then re-split again at sentence
    # boundaries), which would otherwise silently drop a reference_audio
    # assigned here. speaker_id survives both re-splits, so keying by
    # speaker_id and applying it after is what makes this safe.
    reference_clips = reference.extract_reference_clips(
        vocals_path, segments, work_dir,
        raw_audio_path=full_audio, background_path=background_path,
    )

    log.info("Stage 5/8: transcribing (%d speaker segment(s))", len(diarized_segments))
    segments = backends.transcriber.transcribe(vocals_path, diarized_segments)
    _validate_has_speech(segments)

    # Auto-detect likely character/place names from the first pass and
    # re-transcribe with them added as an ASR hint — catches cases like
    # "Yahiko" first coming out as "Yaiko" before the name was known. Only
    # worth a second (~30s) transcription pass if it actually found
    # something new; skips it entirely if auto-detection is unavailable
    # (see stages/auto_names.py) or found nothing beyond what was already
    # explicitly supplied via config.proper_nouns/--names. Re-transcribing
    # against `diarized_segments` (not the first pass's own output) matters
    # so speaker-assignment still refers to the real diarized turns, not an
    # ASR pass standing in for them.
    full_text = " ".join(seg.text for seg in segments)
    detected_names = auto_names.detect_names(full_text)
    new_names = [
        n for n in detected_names
        if n.lower() not in (p.lower() for p in config.proper_nouns)
    ]
    if new_names:
        log.info("auto-detected new name(s), re-transcribing with them as a hint: %s", new_names)
        segments = backends.transcriber.transcribe(vocals_path, diarized_segments, extra_proper_nouns=new_names)
        _validate_has_speech(segments)
    config.proper_nouns = list(config.proper_nouns) + new_names

    # Whisper's own segment boundaries aren't sentence-aware and can cut a
    # sentence in half, which measurably hurts translation quality for the
    # cut sentences (diagnosed from a real run: the two segments split
    # mid-sentence produced the worst translations in the whole file). Regroup
    # into complete sentences before translating.
    segments = resegment.resegment_to_sentences(segments)

    # Fill in each reference clip's transcript from the real word-level
    # transcription now available (words whose time overlaps the clip's
    # window) — needed for zero-shot prosody cloning in the synthesize stage
    # below. Segments themselves still only need reference_audio/_text.
    # Filtering to the clip's own speaker_id matters once there's more than
    # one speaker — without it, a clip could pick up another speaker's words
    # if their timing happens to overlap.
    for speaker_id, clip in reference_clips.items():
        overlapping = [
            w for seg in segments if seg.speaker_id == speaker_id
            for w in seg.words if w.start < clip.end and w.end > clip.start
        ]
        clip.text = " ".join(w.text for w in overlapping).strip()

    # Cross-run voice consistency: match each of this run's speaker-level
    # clips against a persistent bank of previously-recognized voices (see
    # stages/voice_bank.py) so the same character clones from the same
    # reference across separate dubtool runs (e.g. different episodes),
    # instead of each run only ever seeing its own file. Only affects the
    # speaker-level *fallback* clip — per-segment references below are
    # unaffected, so within-scene emotional variety still comes from each
    # line's own audio.
    if config.voice_bank_dir is not None:
        bank = voice_bank.VoiceBank(config.voice_bank_dir, config.voice_bank_similarity_threshold)
        for speaker_id, clip in list(reference_clips.items()):
            score = reference.clip_quality_score(clip.path)
            reference_clips[speaker_id] = bank.find_or_add(clip.path, clip.text, score)

    # Prefer each segment's *own* original audio as its cloning reference
    # over the one fixed speaker-level clip above — a single reference
    # can't represent a character whose delivery actually varies within a
    # scene (calm menace vs. a shouted attack name, say); the segment's own
    # audio already has the right register for its own content, so using it
    # transfers that per-line instead of imposing one moment's style on
    # every line a speaker has. Falls back to the speaker-level clip (and
    # from there, the bundled default) when a segment's own audio is too
    # short or clipped to be a reliable reference on its own.
    for i, seg in enumerate(segments):
        own_clip = reference.extract_segment_reference(
            seg, vocals_path, work_dir / f"segment_ref_{i:03d}_{seg.speaker_id}.wav",
            raw_audio_path=full_audio, background_path=background_path,
        )
        speaker_clip = reference_clips.get(seg.speaker_id)
        clip = own_clip or speaker_clip
        seg.reference_audio = clip.path if clip else config.models.default_reference_audio
        seg.reference_text = clip.text if clip and clip.text else config.models.default_reference_text

    # Classify each segment's likely emotion from its own original audio
    # (see stages/emotion.py) — used below to request a matching delivery
    # style explicitly via inference_instruct2 instead of leaving it purely
    # to zero-shot's implicit transfer, for segments with a strong enough
    # acoustic signal to classify confidently. Skipped when the user has
    # already forced a specific style via --emotion for every segment.
    detected_emotions: dict[int, str] = {} if config.tts_emotion else emotion.classify_segments(segments, vocals_path)

    log.info("Stage 6/8: translating %d segment(s) to %s", len(segments), config.target_language)
    for seg in segments:
        # Prefer the transcriber's own detected language over the config
        # default — config.source_language is normally unset (auto-detect),
        # and the translator needs a real source language to pick the right
        # pivot, not the literal string "auto".
        source_lang = seg.detected_language or config.source_language or "auto"
        protected_text, name_map = proper_nouns.protect(seg.text, config.proper_nouns)
        translated = backends.translator.translate(
            protected_text, source_lang=source_lang, target_lang=config.target_language
        )
        seg.translated_text = proper_nouns.restore(translated, name_map)
        log.debug("  %.2fs-%.2fs: %r -> %r", seg.start, seg.end, seg.text, seg.translated_text)

    log.info("Stage 7/8: synthesizing + time-aligning")
    sr = backends.tts.sample_rate
    for i, seg in enumerate(segments):
        ref, ref_text = seg.reference_audio, seg.reference_text
        # Explicit --emotion always wins; otherwise use this segment's own
        # detected emotion (if confidently classified) to request a
        # matching delivery via inference_instruct2, else None (zero-shot,
        # natural prosody from the reference clip).
        seg_emotion = config.tts_emotion or detected_emotions.get(i)

        # Pass 1: natural pace, just to measure how long this text actually
        # takes to speak. Pass 2: re-synthesize at a native `speed` aimed at
        # seg.duration directly, so the waveform-domain stretch below only
        # has a small residual gap to close instead of carrying the whole
        # correction (see config.py for why — this used to be a single pass
        # and sounded robotic/over-stretched as a result).
        natural = backends.tts.synthesize(
            seg.translated_text, ref, reference_text=ref_text, emotion=seg_emotion
        )
        natural_duration = len(natural) / sr
        if natural_duration <= 0:
            raw = natural
        else:
            native_speed = natural_duration / seg.duration
            native_speed = max(config.min_native_speed, min(config.max_native_speed, native_speed))
            if abs(native_speed - 1.0) < 0.02:
                raw = natural  # already close enough, skip a redundant second pass
            else:
                raw = backends.tts.synthesize(
                    seg.translated_text, ref, reference_text=ref_text,
                    emotion=seg_emotion, speed=native_speed,
                )

        seg.synthesized_audio = align.align_segment(
            raw,
            sr,
            target_duration=seg.duration,
            min_ratio=config.min_stretch_ratio,
            max_ratio=config.max_stretch_ratio,
        )
        seg.synthesized_sr = backends.tts.sample_rate

    log.info("Stage 8/8: mixing" + (" + muxing" if has_video else " (no video to mux into)"))
    final_audio = mix.mix_segments(
        segments, background_path, work_dir / "final_audio.wav", sample_rate=backends.tts.sample_rate
    )
    if has_video:
        mux.mux(video_path, final_audio, output_path)
        actual_output = output_path
    else:
        actual_output = mux.write_audio_only(final_audio, output_path)

    if not config.keep_intermediate:
        _cleanup(work_dir, keep={final_audio})

    return DubbingResult(output_video=actual_output, segments=segments, intermediate_dir=work_dir if config.keep_intermediate else None)


def _validate_has_speech(segments: list[Segment]) -> None:
    if not segments or not any(s.text.strip() for s in segments):
        raise NoSpeechDetectedError(
            "No speech was detected in the input audio. Check that the video "
            "actually contains dialogue and that the vocal-separation step "
            "didn't strip it out."
        )


def _validate_speaker_count(segments: list[Segment], expected: int | None) -> None:
    found = len({s.speaker_id for s in segments})
    if expected is not None and found != expected:
        log.warning("Diarization found %d speaker(s), expected %d", found, expected)
    if found > 12:
        raise UnreasonableSpeakerCountError(
            f"Diarization found {found} distinct speakers, which is almost "
            "certainly a diarization error rather than a real video with "
            "that many speakers. Try passing --num-speakers explicitly."
        )


def _cleanup(work_dir: Path, keep: set[Path]) -> None:
    for p in work_dir.iterdir():
        if p not in keep and p.is_file():
            p.unlink()
