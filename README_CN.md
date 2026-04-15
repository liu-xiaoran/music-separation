# music-sep

[English](README.md)

基于 Demucs 的 AI 音频分离 CLI 工具，集成歌词识别与音频可视化。

## 功能

- **音乐源分离** — 基于 Demucs，将歌曲拆分为 vocals / drums / bass / other 等独立音轨
- **歌词识别** — 基于 faster-whisper，自动识别歌词并输出带时间戳的歌词文件（SRT/VTT/TXT/JSON）
- **音频可视化** — 基于 librosa + matplotlib，生成波形图、频谱图、梅尔频谱图

三大功能独立可选，可单独使用也可组合使用。

## 安装

### 前置要求

- Python >= 3.10
- ffmpeg（系统级安装）
- PyTorch >= 2.0（需自行安装对应平台版本）

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

### 安装 music-sep

```bash
git clone <repo-url> music-separation && cd music-separation
pip install -e .
```

安装完成后即可使用 `music-sep` 命令。

## 快速开始

```bash
# 基础分离（4轨：vocals / drums / bass / other）
music-sep song.mp3

# 仅提取人声 + 伴奏（卡拉OK模式）
music-sep song.mp3 --two-stems vocals

# 分离 + 歌词识别
music-sep song.mp3 --lyrics

# 分离 + 可视化
music-sep song.mp3 --visualize

# 全量输出：分离 + 歌词 + 可视化
music-sep song.mp3 --lyrics --visualize

# 使用 6 轨模型（+ guitar, piano）
music-sep song.mp3 --model htdemucs_6s

# 查看执行计划但不实际运行
music-sep song.mp3 --lyrics --visualize --dry-run

# 使用配置文件
music-sep song.mp3 --config music-sep.toml
```

## 命令参考

### `music-sep separate`（默认命令）

```
music-sep <input> [OPTIONS]
music-sep separate <input> [OPTIONS]
```

#### 分离参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `htdemucs` | Demucs 模型名称 |
| `--device` | `auto` | 计算设备：cpu / cuda / mps / auto |
| `--shifts` | `1` | 随机偏移次数（越多质量越高，速度越慢） |
| `--overlap` | `0.25` | 偏移重叠比例 (0.0 ~ 1.0) |
| `--two-stems` | — | 仅输出指定音轨 + 残差（如 `vocals`） |
| `--stems` | — | 仅输出指定音轨（可多次使用） |

**可用模型：**

| 模型 | 轨数 | 说明 |
|------|------|------|
| `htdemucs` | 4 | drums, bass, other, vocals（默认） |
| `htdemucs_6s` | 6 | + guitar, piano |
| `htdemucs_ft` | 4 | 微调版，质量更高 |
| `mdx` | 4 | MDX 基线模型 |
| `mdx_extra` | 4 | MDX 增强模型 |
| `hdemucs_mmi` | 2 | 仅人声 / 伴奏 |

#### 歌词参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lyrics` / `--no-lyrics` | 禁用 | 启用歌词识别 |
| `--whisper-model` | `medium` | Whisper 模型大小 |
| `--whisper-device` | `auto` | 歌词识别设备（不支持 mps） |
| `--language` | 自动检测 | 强制语言代码（如 zh / en / ja） |
| `--lyrics-format` | `srt` | 歌词格式：srt / lrc / vtt / txt / json |

#### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--visualize` / `--no-visualize` | 禁用 | 启用可视化 |
| `--viz-types` | 全部 | 可视化类型：waveform / spectrogram / mel |

#### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `demo` | 输出根目录 |
| `--format` | `wav` | 输出格式：wav / mp3 / flac |
| `--bitrate` | `128k` | MP3 比特率 |
| `--overwrite` | 否 | 覆盖已有输出目录 |

### `music-sep models`

列出可用的 Demucs 模型及安装状态。

```bash
music-sep models                # 列出所有模型
music-sep models --installed-only   # 仅显示已安装的
```

### `music-sep info`

查看音频文件元信息。

```bash
music-sep info song.mp3
```

## 配置文件

支持 TOML 配置文件，CLI 参数优先级更高。

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

优先级：**CLI 参数 > TOML 配置 > 内置默认值**

## 输出结构

```
demo/
└── song/
    ├── stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   └── other.wav
    ├── lyrics.srt              # --lyrics 时生成（支持 srt/lrc/vtt/txt/json）
    └── visualizations/         # --visualize 时生成
        ├── waveform_original.png
        ├── waveform_vocals.png
        ├── spectrogram_original.png
        └── ...
```

重复运行同一文件时会提示输出已存在，使用 `--overwrite` 覆盖。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码检查
ruff check src/
```

## 许可证

Apache-2.0
