#!/usr/bin/env bash
# Batch-dub a set of episodes one at a time, keeping peak disk use flat.
#
# Each episode is dubbed, then the result is VERIFIED before anything is
# removed: a non-trivial file size, a real video stream, a real audio
# stream, and a duration within tolerance of the source. A dub that fails
# any check leaves the source untouched and the run moves on, so one bad
# episode can't cascade into deleting a source we can't reproduce.
#
# Deleting sources is opt-in via DELETE_SOURCES=1 and never the default:
# the dubbed output carries only video + Uzbek audio, so removing the
# source also discards the English and Japanese audio tracks for good.
#
# Usage:
#   scripts/dub_batch.sh ~/Downloads/"Jujutsu Kaisen - S01E"*.mkv
#   DELETE_SOURCES=1 scripts/dub_batch.sh ...   # only after QC'ing output

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUBTOOL="$REPO/.venv/bin/dubtool"
GLOSSARY="$REPO/assets/jjk_glossary.yaml"
OUTDIR="$REPO/output"
LOGDIR="$REPO/output/logs"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
DELETE_SOURCES="${DELETE_SOURCES:-0}"

mkdir -p "$OUTDIR" "$LOGDIR"

free_gb() { df -g /System/Volumes/Data | awk 'NR==2{print $4}'; }
dur_of()  { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null; }

ok=0; failed=0; skipped=0

for src in "$@"; do
  [ -f "$src" ] || { echo "SKIP (missing): $src"; skipped=$((skipped+1)); continue; }
  base="$(basename "$src")"
  out="$OUTDIR/$base"
  log="$LOGDIR/${base%.*}.log"

  if [ -s "$out" ]; then
    echo "SKIP (already dubbed): $base"; skipped=$((skipped+1)); continue
  fi

  avail="$(free_gb)"
  if [ "$avail" -lt "$MIN_FREE_GB" ]; then
    echo "STOP: only ${avail}GB free, need >=${MIN_FREE_GB}GB. Remaining episodes not attempted."
    break
  fi

  echo "=== Dubbing: $base  (${avail}GB free) ==="
  start=$(date +%s)
  "$DUBTOOL" "$src" --glossary "$GLOSSARY" -o "$out" -v >"$log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [ $rc -ne 0 ]; then
    echo "FAIL (exit $rc) after ${elapsed}s -- see $log"
    rm -f "$out"; failed=$((failed+1)); continue
  fi

  # --- verify before trusting the result -------------------------------
  size=$(stat -f%z "$out" 2>/dev/null || echo 0)
  has_v=$(ffprobe -v error -select_streams v -show_entries stream=codec_type -of csv=p=0 "$out" 2>/dev/null | head -1)
  has_a=$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$out" 2>/dev/null | head -1)
  sdur=$(dur_of "$src"); odur=$(dur_of "$out")
  drift=$(python3 -c "print(abs(${odur:-0}-${sdur:-1}))" 2>/dev/null || echo 999)
  short=$(python3 -c "print(1 if abs(${odur:-0}-${sdur:-1}) > 2.0 else 0)" 2>/dev/null || echo 1)

  if [ "$size" -lt 1000000 ] || [ "$has_v" != "video" ] || [ "$has_a" != "audio" ] || [ "$short" -ne 0 ]; then
    echo "FAIL verification: size=${size}B video=${has_v:-none} audio=${has_a:-none} drift=${drift}s"
    echo "  -> source kept, output left at $out for inspection"
    failed=$((failed+1)); continue
  fi

  printf "OK  %s  (%dm%02ds, %.0fMB, drift %.2fs)\n" \
    "$base" $((elapsed/60)) $((elapsed%60)) "$(echo "$size/1048576" | bc -l)" "$drift"
  ok=$((ok+1))

  if [ "$DELETE_SOURCES" = "1" ]; then
    echo "  -> deleting source (DELETE_SOURCES=1): $src"
    rm -f "$src"
  fi
done

echo ""
echo "=== done: $ok ok, $failed failed, $skipped skipped | $(free_gb)GB free ==="
