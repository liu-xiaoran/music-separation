# music-sep

[中文文档](README_CN.md)

AI-powered audio separation CLI tool built on Demucs, with integrated lyrics transcription and audio visualization.

## Features

- **Music Source Separation** — Powered by Demucs, splits songs into individual stems (vocals / drums / bass / other)
- **Lyrics Transcription** — Powered by faster-whisper, automatically transcribes lyrics with timestamps (SRT/VTT/TXT/JSON)
- **Audio Visualization** — Powered by librosa + matplotlib, generates waveform, spectrogram, and mel spectrogram plots

All three features are independently selectable — use them individually or combined.

## Installation

### Prerequisites

- Python >= 3.10
- ffmpeg (system-level install)
- PyTorch >= 2.0 (install the appropriate version for your platform)

```bash
# macOS (Apple Silicon / Intel)
pip install torch torchaudio

# Linux/Windows + NVIDIA GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Linux/Windows CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```bash
# ffmpeg
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
```

### Install music-sep

```bash
git clone <repo-url> music-separation && cd music-separation
pip install -e .
```

After installation, the `music-sep` command will be available.

## Quick Start

```bash
# Basic separation (4 stems: vocals / drums / bass / other)
music-sep song.mp3

# Extract vocals + accompaniment only (karaoke mode)
music-sep song.mp3 --two-stems vocals

# Separation + lyrics transcription
music-sep song.mp3 --lyrics

# Separation + visualization
music-sep song.mp3 --visualize

# Full output: separation + lyrics + visualization
music-sep song.mp3 --lyrics --visualize

# Use 6-stem model (+ guitar, piano)
music-sep song.mp3 --model htdemucs_6s

# Preview execution plan without running
music-sep song.mp3 --lyrics --visualize --dry-run

# Use a config file
music-sep song.mp3 --config music-sep.toml
```

## Command Reference

### `music-sep separate` (default command)

```
music-sep <input> [OPTIONS]
music-sep separate <input> [OPTIONS]
```

#### Separation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `htdemucs` | Demucs model name |
| `--device` | `auto` | Compute device: cpu / cuda / mps / auto |
| `--shifts` | `1` | Number of random shifts (higher = better quality, slower) |
| `--overlap` | `0.25` | Shift overlap ratio (0.0 ~ 1.0) |
| `--two-stems` | — | Output only the specified stem + residual (e.g. `vocals`) |
| `--stems` | — | Output only specified stems (can be used multiple times) |

**Available models:**

| Model | Stems | Description |
|-------|-------|-------------|
| `htdemucs` | 4 | drums, bass, other, vocals (default) |
| `htdemucs_6s` | 6 | + guitar, piano |
| `htdemucs_ft` | 4 | Fine-tuned version, higher quality |
| `mdx` | 4 | MDX baseline model |
| `mdx_extra` | 4 | MDX enhanced model |
| `hdemucs_mmi` | 2 | Vocals / accompaniment only |

#### Lyrics Options

| Option | Default | Description |
|--------|---------|-------------|
| `--lyrics` / `--no-lyrics` | Disabled | Enable lyrics transcription |
| `--whisper-model` | `medium` | Whisper model size |
| `--whisper-device` | `auto` | Lyrics transcription device (mps not supported) |
| `--language` | Auto-detect | Force language code (e.g. zh / en / ja) |
| `--lyrics-format` | `srt` | Lyrics format: srt / lrc / vtt / txt / json |

#### Visualization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--visualize` / `--no-visualize` | Disabled | Enable visualization |
| `--viz-types` | All | Visualization types: waveform / spectrogram / mel |

#### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `demo` | Output root directory |
| `--format` | `wav` | Output format: wav / mp3 / flac |
| `--bitrate` | `128k` | MP3 bitrate |
| `--overwrite` | No | Overwrite existing output directory |

### `music-sep models`

List available Demucs models and their install status.

```bash
music-sep models                # List all models
music-sep models --installed-only   # Show installed only
```

### `music-sep info`

Display audio file metadata.

```bash
music-sep info song.mp3
```

## Configuration File

Supports TOML config files. CLI arguments take precedence.

```toml
# music-sep.toml

[separation]
model = "htdemucs"
device = "auto"
shifts = 1
overlap = 0.25
# two_stems = "vocals"
# stems = ["vocals", "drums"]

[lyrics]
enabled = false
whisper_model = "medium"
whisper_device = "auto"
# language = "zh"
output_format = "srt"  # srt / lrc / vtt / txt / json

[visualization]
enabled = false
types = ["waveform", "spectrogram", "mel"]

[output]
output_dir = "demo"
format = "wav"
bitrate = "128k"
overwrite = false
```

Priority: **CLI args > TOML config > Built-in defaults**

## Output Structure

```
demo/
└── song/
    ├── stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   └── other.wav
    ├── lyrics.srt              # Generated with --lyrics (supports srt/lrc/vtt/txt/json)
    └── visualizations/         # Generated with --visualize
        ├── waveform_original.png
        ├── waveform_vocals.png
        ├── spectrogram_original.png
        └── ...
```

Running on the same file again will warn that output already exists. Use `--overwrite` to replace.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/
```

## License

Apache-2.0
