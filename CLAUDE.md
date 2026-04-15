# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

music-sep is a CLI tool for AI-powered music source separation, lyrics transcription, and audio visualization. It wraps Demucs (separation), faster-whisper (lyrics), and librosa+matplotlib (visualization) into a unified pipeline.

## Development Environment

Uses conda environment `music-sep` with Python 3.11. Activate before running:

```bash
conda activate music-sep
```

The project uses `hatchling` as the build backend and installs in editable mode.

## Common Commands

```bash
# Install (editable with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_config.py -v

# Run a specific test
pytest tests/test_config.py::test_merge_config_defaults -v

# Lint
ruff check src/

# Type check
mypy src/

# CLI usage examples
music-sep separate song.mp3 --two-stems vocals --overwrite -v
music-sep separate song.mp3 --lyrics --visualize
music-sep models
music-sep info song.mp3
```

## Architecture

The entry point is `cli.py` (Typer app with commands: `separate`, `models`, `info`). The `separate` command delegates to `Pipeline` which orchestrates 4 stages:

1. **Validation & Prep** (`utils.py`) — validates input file, detects device (cuda/mps/cpu), resolves output paths (`outputs.py`)
2. **Separation** (`separation.py`) — uses Demucs low-level API (`demucs.pretrained.get_model` + `demucs.apply.apply_model` + `demucs.separate.load_track`), NOT the `demucs.api` module which doesn't exist in demucs 4.0.x
3. **Lyrics** (`lyrics.py`, optional) — wraps `faster_whisper.WhisperModel`; ctranslate2 backend does NOT support MPS (falls back to CPU). Supported output formats: srt, lrc, vtt, txt, json
4. **Visualization** (`visualization.py`, optional) — generates waveform/spectrogram/mel spectrogram PNGs via librosa + matplotlib (Agg backend)

**Configuration** follows a 3-layer merge: CLI args > TOML file > built-in defaults. All config dataclasses live in `config.py`. Constants (available models, supported formats) in `constants.py`. Custom exceptions in `exceptions.py`.

**Output structure**: `<output_dir>/<song_name>/stems/` for audio, `lyrics.<fmt>` for lyrics, `visualizations/` for PNGs. The `outputs.py` module handles path resolution and directory creation.

## Key Dependencies

- **demucs 4.0.1** — uses `demucs.pretrained`, `demucs.apply`, `demucs.separate` (not `demucs.api`)
- **faster-whisper** — ctranslate2 backend, MPS not supported
- **torchaudio** — requires `torchcodec` for saving audio files in newer versions
- All user-facing strings and log messages are in Chinese
