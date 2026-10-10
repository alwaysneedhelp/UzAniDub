#!/usr/bin/env bash
# Automates OpenVoice V2 setup so voice_clone_test.py can find the
# `openvoice` module and its checkpoints.
#
# No GitHub login or credentials are needed anywhere here -- OpenVoice is
# a public repo, cloned over plain HTTPS.
#
# IMPORTANT: run this with the SAME Python virtual environment active that
# you used for voice_clone_test.py's own requirements.txt, so everything
# (torch, openvoice, MeloTTS, etc.) lands in one place. If you haven't
# activated it yet:
#   source venv/bin/activate
#
# Usage:
#   chmod +x setup_openvoice.sh
#   ./setup_openvoice.sh                     # installs into ~/openvoice_setup
#   ./setup_openvoice.sh /some/other/path    # or a custom location

set -euo pipefail

INSTALL_DIR="${1:-$HOME/openvoice_setup}"
CKPT_URL="https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip"

# --- Preflight: native build deps ---------------------------------------
# MeloTTS (and Demucs) pull in `av` (PyAV), which compiles from source if
# no prebuilt wheel matches your Python/platform. Building it needs
# pkg-config plus FFmpeg's dev libraries on the system -- not the ffmpeg
# CLI you already installed for voice_clone_test.py, the *library* headers.
if ! command -v pkg-config >/dev/null 2>&1; then
  echo "-- pkg-config not found --"
  if command -v brew >/dev/null 2>&1; then
    echo "-- Installing pkg-config and ffmpeg dev libraries via Homebrew --"
    brew install pkg-config ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    echo "-- Installing pkg-config and ffmpeg dev libraries via apt --"
    sudo apt-get update && sudo apt-get install -y pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
  else
    echo "ERROR: pkg-config is missing and I don't see brew or apt-get to install it."
    echo "Install pkg-config and FFmpeg's dev libraries for your system, then rerun this script."
    exit 1
  fi
else
  echo "-- pkg-config found, continuing --"
fi

echo "== Installing OpenVoice into: $INSTALL_DIR =="
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ ! -d "OpenVoice" ]; then
  echo "-- Cloning myshell-ai/OpenVoice (public repo, no login required) --"
  git clone https://github.com/myshell-ai/OpenVoice.git
else
  echo "-- OpenVoice already cloned, skipping --"
fi
cd OpenVoice

# --- Work around a known stale pin in OpenVoice's own requirements.txt --
# OpenVoice pins faster-whisper==0.9.0, which itself demands av==10.* (old
# PyAV). That version has no prebuilt wheel for current Python/macOS, so
# pip tries to compile it from source and fails -- this is the actual
# cause of "Failed to build 'av'", not a missing pkg-config on your end.
# See: https://github.com/myshell-ai/OpenVoice/issues/455
#
# Fix: install a modern, wheel-available av FIRST, then relax the
# faster-whisper pin in requirements.txt so pip doesn't try to downgrade
# it and rebuild the old one.
echo "-- Pre-installing a modern, prebuilt 'av' wheel (sidesteps the stale pin) --"
python3 -m pip install --upgrade pip
python3 -m pip install --only-binary=:all: "av>=11,<15"

if [ -f requirements.txt ] && grep -q "faster-whisper==0.9.0" requirements.txt; then
  echo "-- Relaxing OpenVoice's stale faster-whisper==0.9.0 pin to >=0.10.0 --"
  # -i.bak works on both macOS (BSD sed) and Linux (GNU sed)
  sed -i.bak 's/faster-whisper==0\.9\.0/faster-whisper>=0.10.0/' requirements.txt
fi

echo "-- Installing the openvoice package into your active Python environment --"
python3 -m pip install -e .

echo "-- Installing MeloTTS + its dictionary data (required by OpenVoice V2) --"
python3 -m pip install git+https://github.com/myshell-ai/MeloTTS.git
python3 -m unidic download

if [ ! -f "checkpoints_v2/converter/checkpoint.pth" ]; then
  echo "-- Downloading V2 checkpoints (a few hundred MB, be patient) --"
  curl -L -o checkpoints_v2.zip "$CKPT_URL"
  unzip -o checkpoints_v2.zip
  rm checkpoints_v2.zip
else
  echo "-- Checkpoints already present, skipping download --"
fi

echo ""
echo "== Done. OpenVoice is set up at: $INSTALL_DIR/OpenVoice =="
echo ""
echo "Now either:"
echo "  A) copy voice_clone_test.py into $INSTALL_DIR/OpenVoice and run it"
echo "     from inside that folder, or"
echo "  B) leave voice_clone_test.py where it is and export these first:"
echo ""
echo "       export OPENVOICE_REPO=$INSTALL_DIR/OpenVoice"
echo "       export OPENVOICE_CKPT_V2=$INSTALL_DIR/OpenVoice/checkpoints_v2/converter"
echo ""
echo "Then run voice_clone_test.py as before."