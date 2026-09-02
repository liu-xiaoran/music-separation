# music-sep

[中文文档](README_CN.md)

AI-powered audio separation CLI tool built on Demucs, with integrated lyrics transcription and audio visualization.

## Features

- **Music Source Separation** — Powered by Demucs, splits songs into individual stems (vocals / drums / bass / other)
- **Lyrics Transcription** — Powered by faster-whisper, automatically transcribes lyrics with timestamps (SRT/LRC/VTT/TXT/JSON)
- **Audio Visualization** — Powered by librosa + matplotlib, generates waveform, spectrogram, and mel spectrogram plots

Source separation is the mandatory processing stage. Lyrics transcription and visualization are optional post-processing stages that can be enabled independently or together.

## Installation

### Prerequisites

- Python >= 3.10
- ffmpeg (system-level install; required by some input/output codec paths)
- PyTorch and torchaudio >= 2.0, installed for the target CPU/GPU platform

Install a matching PyTorch/torchaudio pair before installing the project when a platform-specific wheel index is required:

```bash
# macOS (Apple Silicon / Intel)
pip install torch torchaudio

# Linux/Windows + NVIDIA GPU (example CUDA index; choose the index matching your system)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Linux/Windows CPU only
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install ffmpeg with the package manager for your operating system, for example:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Install music-sep

```bash
git clone https://github.com/liu-xiaoran/music-separation.git
cd music-separation
pip install -e .
```

The active development environment for this repository is the Conda environment `music-sep` with Python 3.11:

```bash
conda activate music-sep
pip install -e ".[dev]"
```

After installation, the `music-sep` command is available.

## Quick Start

The current CLI requires the explicit `separate` subcommand.

```bash
# Basic separation (4 stems: vocals / drums / bass / other)
music-sep separate song.mp3

# Extract vocals + accompaniment only (karaoke mode)
music-sep separate song.mp3 --two-stems vocals

# Separation + lyrics transcription
music-sep separate song.mp3 --lyrics

# Separation + visualization
music-sep separate song.mp3 --visualize

# Full output: separation + lyrics + visualization
music-sep separate song.mp3 --lyrics --visualize

# Use 6-stem model (+ guitar, piano)
music-sep separate song.mp3 --model htdemucs_6s

# Display the CLI-level execution plan without running the pipeline
music-sep separate song.mp3 --lyrics --visualize --dry-run

# Use a config file
music-sep separate song.mp3 --config music-sep.toml
```

## Command Reference

### `music-sep separate`

```text
music-sep separate <input> [OPTIONS]
```

The default-command alias `music-sep <input>` is not implemented in the current version. Use the explicit subcommand in scripts and agent automation.

#### Separation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `htdemucs` | Demucs model name |
| `--device` | `auto` | Compute device: cpu / cuda / mps / auto |
| `--shifts` | `1` | Number of random shifts (higher = better quality, slower) |
| `--overlap` | `0.25` | Shift overlap ratio (0.0–1.0) |
| `--two-stems` | — | Output the specified stem plus its residual (for example, `vocals`) |
| `--stems` | — | Output selected stems; repeat the option for multiple stems |

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
| `--lyrics` / `--no-lyrics` | Disabled | Enable or disable lyrics transcription |
| `--whisper-model` | `medium` | Whisper model size |
| `--whisper-device` | `auto` | Requested device: cpu / cuda / mps / auto. CTranslate2 does not run on MPS, so MPS resolves to CPU with a visible fallback reason |
| `--language` | Auto-detect | Force a language code (for example, zh / en / ja) |
| `--lyrics-format` | `srt` | Lyrics format: srt / lrc / vtt / txt / json |

#### Visualization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--visualize` / `--no-visualize` | Disabled | Enable or disable visualization |
| `--viz-types` | All | Visualization type; repeat for waveform / spectrogram / mel |

#### Output and Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `demo` | Output root directory |
| `--format` | `wav` | Output audio format: wav / mp3 / flac |
| `--bitrate` | `128k` | MP3 bitrate (`8k`–`320k`) |
| `--overwrite` / `--no-overwrite` | No | Replace an existing song output directory |
| `--config` | — | TOML configuration file |
| `--dry-run` | No | Print the merged CLI plan and exit before constructing the pipeline |
| `-v`, `--verbose` | No | Enable debug logging |
| `-q`, `--quiet` | No | Show errors only; verbose takes precedence if both are set |

### `music-sep models`

Lists available Demucs models and an approximate, cache-directory-based installed indication.

```bash
music-sep models
music-sep models --installed-only
```

The installed indication is heuristic; it is not a complete checkpoint integrity or availability check. Models may be downloaded by their upstream libraries on first real use.

### `music-sep info`

Displays audio file metadata. The current implementation decodes the audio with librosa and also queries SoundFile metadata, so codec support still depends on the local runtime.

```bash
music-sep info song.mp3
```

## Configuration File

The project supports four TOML tables. Effective values are selected using **CLI non-`None` > TOML non-`None` > built-in default**, and the merged result is validated afterward.

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

Important merge and validation rules:

- Explicit CLI booleans, including `False`, override TOML values.
- `stems` and `types` lists replace lower-priority lists; lists are not concatenated.
- `two_stems` and `stems` are mutually exclusive.
- `shifts` must be a non-boolean integer >= 1.
- `overlap` must be a finite number in the inclusive range 0.0–1.0.
- Unknown TOML tables and fields produce warnings and are ignored.
- Known TOML sections must be tables.
- MP3 bitrate must use `<integer>k` syntax and be between `8k` and `320k`.

## Output Structure

For input `song.mp3` and the default output root:

```text
demo/
└── song/
    ├── stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   └── other.wav
    ├── lyrics.srt              # Written only when --lyrics succeeds
    └── visualizations/         # Directory is prepared for a normal run
        ├── waveform_original.png   # PNGs are written only with --visualize
        ├── waveform_vocals.png
        ├── spectrogram_original.png
        └── ...
```

Supported input filename extensions are `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, and `.wma`. Extension acceptance does not guarantee that every local codec/runtime combination can decode the file.

By default, an existing `<output-dir>/<song>/` directory causes the run to fail. See the overwrite warning in the AI guide before using `--overwrite`.

<!-- AUTO-GENERATED: AI AGENT GUIDE START -->
## AI Coding-Agent Run and Integration Guide

This section is synchronized from `pyproject.toml`, the public compatibility modules under `src/music_sep/`, and the current test suite. It is intended for coding agents that need to run, test, extend, or embed the project without rediscovering its behavioral boundaries.

### 1. Agent Operating Contract

Treat these statements as the current implementation contract:

1. **Separation is mandatory.** The pipeline always performs validation/preparation and Demucs separation. Lyrics and visualization are optional post-processing stages.
2. **Use the explicit CLI form.** Invoke `music-sep separate <input>`; do not assume that `music-sep <input>` is an alias.
3. **Preserve compatibility.** Existing CLI options, TOML fields/defaults/precedence, output paths and filenames, result fields, exceptions, and public module import paths must remain stable unless a change is explicitly documented as a defect fix.
4. **Keep user-facing runtime text in Chinese.** Existing CLI output, diagnostics, and log messages are Chinese by project policy.
5. **Do not bypass validation.** Build configuration through `merge_config()` or pass `AppConfig`/`ValidatedConfig` into `Pipeline`; the pipeline takes its own validated immutable snapshot.
6. **Do not download real models in normal tests.** Use fake models, injectable ports/resolvers, mocks, and deterministic synthetic audio.
7. **Do not publish or commit restricted artifacts.** Never add model weights, model caches, real copyrighted music, vocals, lyrics, transcripts, credentials, or generated user media to Git.

### 2. Reproducible Setup

From the repository root:

```bash
conda activate music-sep
python --version                 # Development target: Python 3.11; package minimum: 3.10
pip install -e ".[dev]"
pip check
music-sep --version
music-sep --help
```

The console script is declared as:

```text
music-sep = music_sep.cli:app
```

`python -m music_sep` reaches the same Typer application, but the explicit `separate` subcommand is still required for processing.

Platform notes:

- Install a mutually compatible `torch`/`torchaudio` pair for the target CPU, CUDA, or MPS environment.
- Demucs is pinned to `4.0.1` and is integrated through its low-level `demucs.pretrained`, `demucs.apply`, and `demucs.separate` APIs. Do not introduce imports from a nonexistent `demucs.api` module.
- WAV/FLAC writing uses SoundFile/libsndfile. MP3 writing uses the Demucs audio path and its encoder dependencies. Input decoding and `info` behavior remain dependent on locally available codecs.
- Model objects are loaded lazily by the Demucs and Whisper implementations, but importing the current pipeline also imports the visualization module and therefore librosa/matplotlib. Do not claim that every heavy dependency is import-lazy.

### 3. Safe CLI Automation

Use a disposable output directory while developing or evaluating changes:

```bash
workdir="$(mktemp -d)"
music-sep separate /path/to/authorized-input.wav --output-dir "$workdir" --dry-run
```

The CLI input argument must exist and be readable before command execution.

Representative real commands:

```bash
# Mandatory separation only
music-sep separate song.wav --output-dir ./tmp-output

# Selected stems; repeat --stems
music-sep separate song.wav \
  --stems vocals --stems drums \
  --output-dir ./tmp-output

# Two-stem target plus residual
music-sep separate song.wav \
  --two-stems vocals \
  --output-dir ./tmp-output

# Optional lyrics and visualization
music-sep separate song.wav \
  --lyrics --lyrics-format lrc \
  --visualize --viz-types waveform --viz-types mel \
  --output-dir ./tmp-output
```

Current `--dry-run` limitation:

- The CLI merges and validates static configuration, prints its plan, and exits **before** constructing `Pipeline`.
- It does not resolve runtime devices, load models, validate model-specific stem names, validate codec availability, check output conflicts through the pipeline, create directories, or write files.
- Therefore, a successful dry-run is not proof that a real run will succeed.

### 4. Configuration and Immutable Boundaries

The compatibility configuration objects in `music_sep.config` are mutable dataclasses:

- `AppConfig`
- `SeparationConfig`
- `LyricsConfig`
- `VisualizationConfig`
- `OutputConfig`

The domain execution boundary uses immutable, slotted dataclasses:

- `RawConfig`: precedence has been applied, but values are not yet trusted.
- `ValidatedConfig`: canonical logical configuration; requested devices may still be `auto`.
- `RuntimeResolution`: requested device, actual device, backend, and optional fallback reason.
- `ResolvedConfig`: validated request plus the stage-specific runtime resolutions for one run.

Use the conversion functions instead of copying fields manually:

```python
from pathlib import Path

from music_sep.config import merge_config, to_app_config, to_validated_config

app_config = merge_config(
    toml_path=Path("music-sep.toml"),
    device="auto",
    lyrics_enabled=True,
    output_dir=Path("tmp-output"),
    overwrite=False,
)

validated = to_validated_config(app_config)
compatibility_copy = to_app_config(validated)
```

Both conversions detach list values. `Pipeline` also validates a directly supplied `ValidatedConfig` again rather than trusting type annotations or caller-owned state.

### 5. Python Pipeline Integration

The package root exports only `__version__`; import operational APIs from their documented modules.

```python
from pathlib import Path

from music_sep.config import merge_config
from music_sep.pipeline import Pipeline

config = merge_config(
    model="htdemucs",
    device="auto",
    lyrics_enabled=False,
    visualize_enabled=False,
    output_dir=Path("tmp-output"),
    overwrite=False,
)

pipeline = Pipeline(config)
result = pipeline.run(Path("song.wav"))

print(result.output_paths.base_dir)
print(result.stems)                  # dict[str, Path]
print(result.lyrics_path)            # Path | None
print(result.visualization_paths)    # list[Path] | None
print(pipeline.resolved_config)      # ResolvedConfig | None
```

Current result fields are:

```text
PipelineResult(
    input_file,
    output_paths,
    stems={},
    lyrics_path=None,
    visualization_paths=None,
)
```

Run semantics relevant to callers:

- The pipeline copies and validates configuration at construction time; it does not mutate the caller's `AppConfig`.
- Separation device resolution is fresh for every `run()`.
- Lyrics device resolution is deferred until mandatory separation succeeds.
- Lyrics uses the generated `vocals` stem when present and falls back to the original input otherwise.
- Lyrics and visualization exceptions are currently logged and swallowed independently. A completed call can therefore represent separation success with missing optional artifacts.
- If lyrics device resolution/transcription fails, visualization can still run.
- `resolved_config` is reset at the beginning of each run to prevent stale state. When lyrics are enabled, a dry-run or a failure before successful lyrics-device resolution can leave it as `None` because a complete snapshot requires that resolution.

For lower-level compatibility integration, current public module APIs include:

```python
from music_sep.lyrics import LyricsTranscriber, TranscriptionResult
from music_sep.outputs import OutputPaths, ensure_output_dirs, resolve_output_paths
from music_sep.separation import SeparationEngine
from music_sep.utils import detect_device, resolve_device, validate_input_file
from music_sep.visualization import Visualizer
```

`SeparationEngine` additionally accepts injected `Separator` and `AudioWriter` implementations. `Pipeline` accepts an injected `RuntimeResolver`. Prefer these boundaries in tests instead of patching private fields or invoking third-party libraries directly.

### 6. Runtime Device Matrix

Device requests are represented separately from actual backend devices.

| Stage/backend | `auto` priority | Explicit MPS | Explicit unavailable CUDA/MPS |
|---------------|-----------------|--------------|-------------------------------|
| Demucs / PyTorch | CUDA → MPS → CPU | Uses MPS when PyTorch reports it available | Raises a device error |
| Whisper / CTranslate2 | CTranslate2 CUDA → CPU | Resolves to CPU and records/logs a fallback reason | CUDA raises if CTranslate2 cannot use CUDA; MPS falls back to CPU |

Important integration rules:

- CTranslate2 CUDA capability is probed with CTranslate2's own device API, not `torch.cuda.is_available()`.
- CTranslate2 never runs on MPS in this project.
- Whisper uses `float16` on CUDA and `int8` on CPU.
- An unavailable optional Whisper device must not prevent mandatory Demucs separation from completing.
- Use `resolve_device()` when the requested and actual devices plus fallback reason are needed. `detect_device()` is the legacy string-only compatibility API.

Example without model construction:

```python
from music_sep.utils import resolve_device

separation_runtime = resolve_device("auto", backend="torch")
lyrics_runtime = resolve_device("mps", backend="ctranslate2")

assert lyrics_runtime.actual == "cpu"
assert lyrics_runtime.fallback_reason is not None
```

### 7. Outputs, Side Effects, and Overwrite Safety

Output paths are derived from `input_file.stem`:

```text
<output_dir>/<song_name>/
├── stems/<stem>.<wav|flac|mp3>
├── lyrics.<srt|lrc|vtt|txt|json>
└── visualizations/<type>_<original|stem>.png
```

Individual stem files are encoded through a temporary same-parent location and then replaced into place. **The song directory as a whole is not transactional yet.** Current `overwrite=True` behavior recursively deletes the existing final song directory before separation begins. If a later mandatory or optional operation fails, the old directory is not restored.

Agent safety rules until transactional publication is implemented:

1. Do not use `--overwrite` on an output directory containing data that must be retained.
2. Prefer a fresh temporary `--output-dir` for every autonomous or test run.
3. If replacement is required, copy or back up the old output outside the target tree first and report the operation to the user; do not silently delete it.
4. Do not call `ensure_output_dirs(..., overwrite=True)` from automation without the same protections.
5. Treat symlink, concurrency, interruption recovery, directory rollback, and transaction locking as not yet implemented.

### 8. Verification Commands

Run commands from the repository root in the activated development environment:

```bash
# Full and non-slow test gates
pytest tests/ -v
pytest -m "not slow"

# Focused examples
pytest tests/test_config.py -v
pytest tests/test_config.py::test_merge_config_defaults -v

# Static quality gates
ruff format --check src tests
ruff check src tests
mypy src

# Dependency and package gates
pip check
python -m build

# Basic import/entry-point smoke
python -c "import music_sep; import music_sep.cli; print(music_sep.__version__)"
music-sep --version
music-sep --help
```

Test-writing constraints:

- Generate short audio deterministically with a fixed seed or use existing synthetic fixtures.
- Inject fake separators, writers, Whisper models, and runtime capability probes.
- Mark tests requiring real model downloads or real hardware as `slow`; do not make them part of the normal offline gate.
- Never make pass/fail assertions based on the development machine's actual CUDA or MPS availability.
- Do not use copyrighted audio or lyrics as committed fixtures.

### 9. Change Map for Agents

Inspect these files before changing the corresponding behavior:

| Concern | Source of truth |
|---------|-----------------|
| Package requirements, entry point, test marker | `pyproject.toml` |
| CLI commands, options, dry-run, exit mapping | `src/music_sep/cli.py` |
| Defaults, TOML schema, precedence, compatibility DTOs | `src/music_sep/config.py` |
| Immutable config and runtime invariants | `src/music_sep/domain/config.py` |
| Models, formats, extensions, device catalogs | `src/music_sep/domain/catalog.py` |
| Device capability and fallback rules | `src/music_sep/adapters/torch_runtime.py` |
| Pipeline stage order and optional failure behavior | `src/music_sep/pipeline.py` |
| Demucs compatibility facade and injected ports | `src/music_sep/separation.py` |
| Demucs implementation | `src/music_sep/adapters/demucs_separator.py` |
| Audio encoding and per-file atomic publication | `src/music_sep/adapters/torchaudio_writer.py` |
| Lyrics model and five output writers | `src/music_sep/lyrics.py` |
| Visualization behavior and naming | `src/music_sep/visualization.py` |
| Output layout and destructive directory overwrite | `src/music_sep/outputs.py` |
| Input validation and compatibility device APIs | `src/music_sep/utils.py` |
| Full behavior/risk catalog for refactoring | `TECHNICAL_HANDOFF.md` |

Architectural direction for new internal code:

```text
domain <- ports <- adapters
                <- compatibility facades / current Pipeline
```

Keep third-party-independent values and invariants in `domain`, behavior contracts in `ports`, and Demucs/Torch/encoding implementations in `adapters`. Existing root modules are compatibility facades and must retain their import paths.

### 10. Completion Checklist

Before declaring an agent change complete:

- [ ] The requested behavior is covered by deterministic tests.
- [ ] No normal test downloads a model or requires GPU/MPS hardware.
- [ ] `pytest tests/ -v` and `pytest -m "not slow"` pass.
- [ ] Ruff format/check and mypy pass.
- [ ] Package build and import smoke pass when packaging or public imports changed.
- [ ] CLI/TOML/output/Python compatibility was preserved or the deliberate defect fix was documented.
- [ ] User-visible CLI/log text remains Chinese.
- [ ] Existing outputs were not destructively overwritten.
- [ ] No model weights, caches, credentials, real copyrighted media, lyrics, or transcripts were added.
- [ ] `README.md` and `TECHNICAL_HANDOFF.md` were updated if observable behavior or a documented risk changed.
- [ ] Unrelated untracked files were left untouched.

<!-- AUTO-GENERATED: AI AGENT GUIDE END -->

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Exclude optional slow/model/hardware tests
pytest -m "not slow"

# Format, lint, and type-check
ruff format --check src tests
ruff check src tests
mypy src

# Build distribution artifacts locally
python -m build
```

## License

Apache-2.0
