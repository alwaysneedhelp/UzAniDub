"""Orchestrates the dubbing pipeline. Talks only to `interfaces.py` shapes and
the plain-function stages (extract/align/mix/mux/reference aren't pluggable —
they're always ffmpeg/numpy/signal-processing heuristics, not swappable
models — so they're functions, not backend classes).

Voice cloning (stages/reference.py, stages/voice_bank.py) is on by
default via config.clone_voices, disable with --no-clone-voices for the
one flat bundled generic Navoiy TTS reference voice instead. Name/term
consistency (stages/glossary.py) is unrelated to cloning and always
active, as is emotion classification (stages/emotion.py) — see that
module for why it now always assigns a delivery style rather than only
for acoustically extreme segments.
"""
from __future__ import annotations

import gc
import logging
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dubtool.config import DubConfig
from dubtool.interfaces import Diarizer, Separator, Transcriber, Translator, TTSBackend
from dubtool.stages import (
    align,
    auto_names,
    emotion,
    extract,
    glossary,
    idioms,
    loudness,
    mix,
    mux,
    reference,
    resegment,
    takes,
    timing,
    voice_bank,
)
from dubtool.stages import glossary as glossary_stage
from dubtool.types import DubbingResult, ReferenceClip, Segment

log = logging.getLogger("dubtool.pipeline")


_BACKEND_NAMES = ("separator", "diarizer", "transcriber", "translator", "tts")


def _release_torch_caches() -> None:
    """Hand back any allocator-cached blocks after a backend is dropped.
    No-op on a pure-CPU install; cheap enough not to bother checking which
    of these a given machine actually has."""
    try:
        import torch
    except ImportError:
        return
    for accel in ("mps", "cuda"):
        backend = getattr(torch, accel, None)
        empty = getattr(backend, "empty_cache", None)
        if empty is not None:
            try:
                empty()
            except Exception:  # pragma: no cover - best-effort only
                pass


class Backends:
    """The five swappable backends, built on first use and releasable.

    Each entry is a zero-argument factory rather than a ready instance,
    because `run()` below uses the five strictly one stage at a time and
    never needs two at once. Constructing them all upfront pinned every
    model's weights for the whole episode — ~12GB for CosyVoice2 + navoiy
    + MADLAD + Whisper + Demucs — which on a 16GB machine pushed the run
    into swap partway through synthesis and took CosyVoice2 from RTF ~3 to
    RTF ~117. Loading each one when its stage starts and dropping it when
    that stage ends holds peak memory to roughly the largest single model
    instead of their sum.

    Use `of()` when you already have ready instances (tests, or any caller
    doing its own lifecycle management) — nothing is lazy or freed then.
    """

    __slots__ = ("_factories", "_built")

    def __init__(
        self,
        *,
        separator: Callable[[], Separator],
        diarizer: Callable[[], Diarizer],
        transcriber: Callable[[], Transcriber],
        translator: Callable[[], Translator],
        tts: Callable[[], TTSBackend],
    ) -> None:
        self._factories: dict[str, Callable[[], object]] = {
            "separator": separator,
            "diarizer": diarizer,
            "transcriber": transcriber,
            "translator": translator,
            "tts": tts,
        }
        self._built: dict[str, object] = {}

    @classmethod
    def of(
        cls,
        *,
        separator: Separator,
        diarizer: Diarizer,
        transcriber: Transcriber,
        translator: Translator,
        tts: TTSBackend,
    ) -> "Backends":
        """Wrap already-constructed backends. `release()` becomes a no-op:
        the caller owns these instances, so dropping our reference wouldn't
        free them and re-accessing must not hand back a different object."""
        instances = {
            "separator": separator, "diarizer": diarizer,
            "transcriber": transcriber, "translator": translator, "tts": tts,
        }
        self = cls(**{name: (lambda v=value: v) for name, value in instances.items()})
        self._built.update(instances)
        return self

    def _get(self, name: str) -> object:
        if name not in self._built:
            log.info("loading %s backend", name)
            self._built[name] = self._factories[name]()
        return self._built[name]

    @property
    def separator(self) -> Separator:
        return self._get("separator")  # type: ignore[return-value]

    @property
    def diarizer(self) -> Diarizer:
        return self._get("diarizer")  # type: ignore[return-value]

    @property
    def transcriber(self) -> Transcriber:
        return self._get("transcriber")  # type: ignore[return-value]

    @property
    def translator(self) -> Translator:
        return self._get("translator")  # type: ignore[return-value]

    @property
    def tts(self) -> TTSBackend:
        return self._get("tts")  # type: ignore[return-value]

    def release(self, *names: str) -> None:
        """Drop the named backends so their weights can be collected before
        the next stage loads its own. Re-accessing one rebuilds it from its
        factory, so this is safe to call even if a later stage needs it
        again — it just pays the load cost twice."""
        unknown = set(names) - set(_BACKEND_NAMES)
        if unknown:
            raise ValueError(f"unknown backend(s): {sorted(unknown)}")
        freed = [name for name in names if self._built.pop(name, None) is not None]
        if freed:
            gc.collect()
            _release_torch_caches()
            log.info("released %s backend(s)", ", ".join(freed))


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
    backends.release("separator")

    log.info("Stage 3/8: diarizing")
    diarized_segments = backends.diarizer.diarize(vocals_path, num_speakers=config.num_speakers)
    backends.release("diarizer")
    _validate_speaker_count(diarized_segments, config.num_speakers)
    segments = diarized_segments

    reference_clips: dict[str, ReferenceClip] = {}
    if config.clone_voices:
        log.info("Stage 4/8: extracting per-speaker reference clips for voice cloning")
        # Computed from the diarized (coarse) segments' speaker_ids, but
        # *applied* after transcribe()+resegment() below — both return fresh
        # Segment lists (re-split at ASR boundaries, then re-split again at
        # sentence boundaries), which would otherwise silently drop a
        # reference_audio assigned here. speaker_id survives both re-splits,
        # so keying by speaker_id and applying it after is what makes this
        # safe.
        reference_clips = reference.extract_reference_clips(
            vocals_path, segments, work_dir,
            raw_audio_path=full_audio, background_path=background_path,
        )
    else:
        log.info("Stage 4/8: skipped (voice cloning disabled — using the bundled generic reference voice)")

    log.info("Stage 5/8: transcribing (%d speaker segment(s))", len(diarized_segments))
    segments = backends.transcriber.transcribe(vocals_path, diarized_segments)
    _validate_has_speech(segments)

    # Auto-detect likely character/place names from the first pass and
    # re-transcribe with them added as an ASR hint — catches cases like
    # "Yahiko" first coming out as "Yaiko" before the name was known. Only
    # worth a second (~30s) transcription pass if it actually found
    # something new; skips it entirely if auto-detection is unavailable
    # (see stages/auto_names.py) or found nothing beyond what was already
    # explicitly supplied via config.glossary/--names/--glossary. Re-
    # transcribing against `diarized_segments` (not the first pass's own
    # output) matters so speaker-assignment still refers to the real
    # diarized turns, not an ASR pass standing in for them.
    full_text = " ".join(seg.text for seg in segments)
    detected_names = auto_names.detect_names(full_text)
    new_names = _names_not_in_glossary(detected_names, config.glossary)
    if new_names:
        log.info("auto-detected new name(s), re-transcribing with them as a hint: %s", new_names)
        segments = backends.transcriber.transcribe(vocals_path, diarized_segments, extra_proper_nouns=new_names)
        _validate_has_speech(segments)
    # An auto-detected name has no known translation, so it's an identity
    # entry (same mechanism as a manually-supplied --names entry) — see
    # stages/glossary.py.
    for name in new_names:
        config.glossary[name] = name
    backends.release("transcriber")

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

    if config.clone_voices:
        # Cross-run voice consistency: match each of this run's speaker-level
        # clips against a persistent bank of previously-recognized voices
        # (see stages/voice_bank.py) so the same character clones from the
        # same reference across separate dubtool runs (e.g. different
        # episodes), instead of each run only ever seeing its own file. Only
        # affects the speaker-level *fallback* clip — per-segment references
        # below are unaffected, so within-scene emotional variety still
        # comes from each line's own audio.
        if config.voice_bank_dir is not None:
            bank = voice_bank.VoiceBank(config.voice_bank_dir, config.voice_bank_similarity_threshold)
            for speaker_id, clip in list(reference_clips.items()):
                score = reference.clip_quality_score(clip.path)
                reference_clips[speaker_id] = bank.find_or_add(clip.path, clip.text, score)

        # One reference per character, used for every line they speak.
        #
        # This used to prefer each line's *own* audio, so that a line copied
        # the delivery its original had. But delivery now comes from the
        # emotion preset (inference_instruct2 doesn't feed the reference to
        # the LLM at all), so the reference only sets the voice -- and a
        # line's own 1-2 seconds is a weak voice sample. Measured on a real
        # clip, two *different* characters came out nearly as similar as two
        # lines of the same one (0.672 vs 0.713 voice similarity, against
        # 0.558 vs 0.652 for the original actors): everyone drifting toward
        # the model's default voice. A character's longest clean stretch, or
        # failing that all their voiced lines joined, is a far stronger
        # sample and the same one every time.
        #
        # Order: the speaker-level clip (matched across episodes by the voice
        # bank above), then all of the character's voiced lines joined, then
        # the line's own audio, then the nearest validated real voice.
        character_refs: dict[str, ReferenceClip | None] = {}
        for i, seg in enumerate(segments):
            if seg.speaker_id not in character_refs:
                clip = reference_clips.get(seg.speaker_id)
                if clip is None:
                    clip = reference.speaker_montage(
                        seg.speaker_id, segments, vocals_path,
                        work_dir / f"montage_{seg.speaker_id}.wav",
                        raw_audio_path=full_audio, background_path=background_path,
                    )
                character_refs[seg.speaker_id] = clip
            clip = character_refs[seg.speaker_id]
            if clip is None:
                clip = reference.extract_segment_reference(
                    seg, vocals_path, work_dir / f"segment_ref_{i:03d}_{seg.speaker_id}.wav",
                    raw_audio_path=full_audio, background_path=background_path,
                    pad_bounds=_same_speaker_bounds(segments, i),
                )
            seg.reference_audio = clip.path if clip else config.models.default_reference_audio
            seg.reference_text = clip.text if clip and clip.text else config.models.default_reference_text
        for speaker_id, clip in character_refs.items():
            if clip is not None:
                log.info("%s: cloning every line from one %.1fs reference", speaker_id, _clip_seconds(clip.path))
        _borrow_nearest_human_reference(segments, config.models.default_reference_audio)
    else:
        # --no-clone-voices: every segment uses the one bundled generic
        # reference voice instead of a per-speaker/per-segment clone.
        for seg in segments:
            seg.reference_audio = config.models.default_reference_audio
            seg.reference_text = config.models.default_reference_text

    # Classify each segment's likely emotion from its own original audio
    # (see stages/emotion.py) — routes almost every segment through
    # inference_instruct2 with an explicit named delivery style (still
    # cloning voice *identity* from seg.reference_audio set above) rather
    # than zero-shot, which turned out to be the actual source of "flat"
    # output, not voice cloning itself — see stages/emotion.py. Skipped
    # when the user has already forced one specific style via --emotion
    # for every segment.
    detected_emotions: dict[int, str] = {} if config.tts_emotion else emotion.classify_segments(segments, vocals_path)

    log.info("Stage 6/8: translating %d segment(s) to %s", len(segments), config.target_language)
    for seg in segments:
        # Prefer the transcriber's own detected language over the config
        # default — config.source_language is normally unset (auto-detect),
        # and the translator needs a real source language to pick the right
        # pivot, not the literal string "auto".
        source_lang = seg.detected_language or config.source_language or "auto"
        # Rewrite common English slang/idioms into plainer English before
        # the MT model sees them — see stages/idioms.py for why this is a
        # cheaper fix than swapping the translation backend entirely. Skipped
        # for a backend that handles idioms itself: the replacements are
        # worded to survive a literal NMT model, which makes them worse input
        # than the original line for a model being asked for natural speech.
        if getattr(backends.translator, "handles_idioms", False):
            normalized_text = seg.text
        else:
            normalized_text = idioms.normalize(seg.text, source_lang)
        protected_text, term_map = glossary.protect(normalized_text, config.glossary)
        translated = backends.translator.translate(
            protected_text, source_lang=source_lang, target_lang=config.target_language,
            # How long this line may run for. Backends that can't act on it
            # ignore it; one that can is asked to fit, which is cheaper than
            # letting mix.py cut the overrun off mid-word.
            target_duration=seg.duration,
        )
        seg.translated_text = glossary.restore(translated, term_map)
        log.debug("  %.2fs-%.2fs: %r -> %r", seg.start, seg.end, seg.text, seg.translated_text)
    backends.release("translator")

    log.info("Stage 7/8: synthesizing + timing")
    sr = backends.tts.sample_rate
    # Where the source audio ends: the last line's room runs to here.
    timeline_end = extract.audio_duration(full_audio)
    stretch_enabled = config.min_stretch_ratio < 1.0 or config.max_stretch_ratio > 1.0
    previous_end = float("-inf")
    late_lines, worst_delay = 0, 0.0
    # Each line's loudness follows the actor's delivery of it -- a shout
    # lands as a shout -- instead of one of a few fixed levels within 2dB of
    # each other (see stages/loudness.py).
    actor_levels = loudness.actor_targets(segments, vocals_path)
    # Intensity: how hard the actor delivered each line, in dB over the
    # scene's median line. Drives vocal effort below, so a scream gets a
    # scream's strained edge rather than only more volume.
    measured_levels = [lv for lv in actor_levels if lv is not None]
    scene_level = float(np.median(measured_levels)) if measured_levels else 0.0
    intensity_db = [
        20 * np.log10(lv / scene_level) if lv is not None and scene_level > 0 else 0.0
        for lv in actor_levels
    ]
    for i, seg in enumerate(segments):
        ref, ref_text = seg.reference_audio, seg.reference_text
        # Explicit --emotion always wins; otherwise use this segment's own
        # detected emotion to request a matching delivery via
        # inference_instruct2 (see stages/emotion.py), else None
        # (zero-shot, natural prosody from the reference clip).
        seg_emotion = config.tts_emotion or detected_emotions.get(i)

        # When this line will really start, and how long it has before the
        # next one is due -- its own slot *plus* the pause after it, which
        # the old fit-to-own-slot approach never used. See stages/timing.py.
        start = timing.place(seg.start, previous_end, config.min_line_gap)
        next_due = segments[i + 1].start if i + 1 < len(segments) else timeline_end
        available = timing.room(start, next_due)

        # Pass 1 at natural pace measures how long the line wants; pass 2
        # re-synthesizes at a native speed only when it has to. Speed comes
        # from CosyVoice2 itself (mel interpolation before vocoding), which
        # regenerates the speech -- never from stretching finished audio.
        natural = _synthesize_take(backends.tts, sr, seg, ref, ref_text, seg_emotion)
        natural_duration = len(natural) / sr
        cap = timing.speed_cap(
            start - seg.start, natural_duration, available,
            config.max_native_speed, config.max_catchup_speed, config.catchup_after,
        )
        speed = timing.choose_speed(
            natural_duration, available, seg.duration, config.min_native_speed, cap,
        )
        if abs(speed - 1.0) < 0.02:
            raw = natural
        else:
            raw = _synthesize_take(backends.tts, sr, seg, ref, ref_text, seg_emotion, speed=speed)
        if stretch_enabled:
            # Opt-in only: a phase vocoder is what made lines sound metallic,
            # so it is off unless a config asks for it for tight lip-sync.
            raw = align.align_segment(
                raw, sr, target_duration=seg.duration,
                min_ratio=config.min_stretch_ratio, max_ratio=config.max_stretch_ratio,
            )

        # CosyVoice2's output level varies take to take for no reason to do
        # with the content, so every line is levelled -- to the actor's level
        # for that line where it can be measured, else to an emotion preset.
        effort = loudness.effort_for(intensity_db[i])
        raw = loudness.vocal_effort(raw, sr, effort)
        target = actor_levels[i]
        seg.synthesized_audio = (
            loudness.match_level(raw, sr, target) if target is not None
            else loudness.normalize(raw, emotion=seg_emotion)
        )
        seg.synthesized_sr = sr
        seg.placed_start = start
        spoken = len(raw) / sr
        previous_end = start + spoken

        delay = start - seg.start
        if delay > 0.05:
            late_lines += 1
            worst_delay = max(worst_delay, delay)
        log.info(
            "line %d at %.2fs%s: %.2fs spoken in %.2fs of room at %.2fx (%s, %s, intensity %+.1fdB%s)",
            i, start, f" (+{delay:.2f}s, waited for the previous line)" if delay > 0.05 else "",
            spoken, available, speed, seg_emotion or "zero-shot", seg.speaker_id, intensity_db[i],
            f", effort +{effort:.1f}dB" if effort > 0.05 else "",
        )
    if late_lines:
        log.info(
            "%d of %d lines started late to let the line before finish; worst delay %.2fs",
            late_lines, len(segments), worst_delay,
        )
    backends.release("tts")

    log.info("Stage 8/8: mixing" + (" + muxing" if has_video else " (no video to mux into)"))
    final_audio = mix.mix_segments(
        segments, background_path, work_dir / "final_audio.wav", sample_rate=sr,
        min_gap=config.min_line_gap,
    )
    if has_video:
        mux.mux(video_path, final_audio, output_path)
        actual_output = output_path
    else:
        actual_output = mux.write_audio_only(final_audio, output_path)

    if not config.keep_intermediate:
        _cleanup(work_dir, keep={final_audio})

    return DubbingResult(output_video=actual_output, segments=segments, intermediate_dir=work_dir if config.keep_intermediate else None)


_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _names_not_in_glossary(detected: list[str], glossary: dict[str, str]) -> list[str]:
    """Auto-detected names worth adding, as pass-through glossary entries.

    A detected name loses a leading article, and is dropped if it overlaps
    a term the glossary already has. Detection returned "the Reverse Cursed
    Technique"; the article kept it from matching the existing entry, so it
    was added as a new verbatim name -- and being the longer term, it then
    won over the real translation, leaving English mid-sentence in the dub:
    "...barcha e'tiborimni the Reverse Cursed Technique ga qaratdim"."""
    kept: list[str] = []
    for name in detected:
        name = _LEADING_ARTICLE.sub("", name).strip()
        if not name:
            continue
        overlaps = any(
            glossary_stage.term_pattern(term).search(name) or glossary_stage.term_pattern(name).search(term)
            for term in glossary
        )
        if not overlaps and name.lower() not in (k.lower() for k in kept):
            kept.append(name)
    return kept


def _synthesize_take(
    tts: TTSBackend,
    sr: int,
    seg: Segment,
    reference_audio: Path,
    reference_text: str,
    emotion: str | None,
    speed: float = 1.0,
    attempts: int = 3,
) -> np.ndarray:
    """Synthesize a line, re-rolling a take that contains no speech.

    The last attempt drops a non-calm preset back to calm: the broken take
    measured on real output was an emotional preset on a two-word line, and
    a plainer reading is better than a line that plays as noise. If every
    attempt fails, the most speech-like one is used (see stages/takes.py).
    """
    best, best_score = None, (-1.0, -1.0)
    for attempt in range(attempts):
        style = emotion
        if attempt == attempts - 1 and emotion not in (None, "calm", "whispers"):
            style = "calm"
        audio = tts.synthesize(
            seg.translated_text, reference_audio, reference_text=reference_text,
            emotion=style, speed=speed,
        )
        usable, voiced = takes.is_usable(audio, sr, style)
        if usable:
            if attempt:
                log.info("%.2fs: take %d is clean (%.0f%% voiced, %s)",
                         seg.start, attempt + 1, voiced * 100, style or "zero-shot")
            return audio
        log.warning(
            "%.2fs: take %d of %r has almost no speech (%.0f%% voiced); re-rolling",
            seg.start, attempt + 1, seg.translated_text[:40], voiced * 100,
        )
        # Ties on voicing go to the take with sound in it: an unvoiced take
        # is often breathy speech, which beats a line that vanishes.
        score = (voiced, float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0)
        if score > best_score:
            best, best_score = audio, score
    log.warning("%.2fs: no clean take in %d attempts; using the best (%.0f%% voiced)",
                seg.start, attempts, best_score[0] * 100)
    return best


def _clip_seconds(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def _borrow_nearest_human_reference(segments: list[Segment], default_audio: Path) -> None:
    """Last resort before the bundled default voice: a line whose speaker has
    no usable audio at all clones the nearest line that does.

    That may be the wrong character, which is why it is the last step. But
    a speaker this short of audio is almost always a fragment diarization
    split off from a neighbour, and the alternative is the bundled reference
    -- a synthetic macOS voice, which is wrong for *every* character and
    makes the line sound like a screen reader."""
    human = [i for i, seg in enumerate(segments) if seg.reference_audio != default_audio]
    if not human:
        return
    for i, seg in enumerate(segments):
        if seg.reference_audio != default_audio:
            continue
        nearest = min(human, key=lambda j: abs(segments[j].start - seg.start))
        donor = segments[nearest]
        log.warning(
            "%.2fs (%s): no usable audio of this speaker; cloning the nearest real voice "
            "(%.2fs, %s) instead of the bundled synthetic one",
            seg.start, seg.speaker_id, donor.start, donor.speaker_id,
        )
        seg.reference_audio = donor.reference_audio
        seg.reference_text = donor.reference_text


def _same_speaker_bounds(segments: list[Segment], index: int) -> tuple[float, float]:
    """Start/end of the unbroken run of segments around `index` that share
    its speaker — the region a reference window may widen into without
    picking up a different voice."""
    speaker = segments[index].speaker_id
    first = index
    while first > 0 and segments[first - 1].speaker_id == speaker:
        first -= 1
    last = index
    while last + 1 < len(segments) and segments[last + 1].speaker_id == speaker:
        last += 1
    return segments[first].start, segments[last].end


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
