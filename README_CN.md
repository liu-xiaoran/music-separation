# music-sep

[English](README.md)

基于 Demucs 的 AI 音频分离 CLI 工具，集成歌词识别与音频可视化。

## 功能

- **音乐源分离** — 基于 Demucs，将歌曲拆分为 vocals / drums / bass / other 等独立音轨
- **歌词识别** — 基于 faster-whisper，自动识别歌词并输出带时间戳的歌词文件（SRT/LRC/VTT/TXT/JSON）
- **音频可视化** — 基于 librosa + matplotlib，生成波形图、频谱图和梅尔频谱图

音乐源分离是必需的处理阶段。歌词识别和音频可视化是可选的后处理阶段，可以分别启用，也可以同时启用。

## 安装

### 前置要求

- Python >= 3.10
- ffmpeg（系统级安装；部分输入/输出编解码路径需要）
- 针对目标 CPU/GPU 平台安装的 PyTorch 和 torchaudio >= 2.0

如果目标平台需要专用 wheel 索引，请先安装相互匹配的 PyTorch/torchaudio：

```bash
# macOS（Apple Silicon / Intel）
pip install torch torchaudio

# Linux/Windows + NVIDIA GPU（CUDA 索引示例；请选择与系统匹配的索引）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Linux/Windows 纯 CPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

使用操作系统的包管理器安装 ffmpeg，例如：

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 安装 music-sep

```bash
git clone https://github.com/liu-xiaoran/music-separation.git
cd music-separation
pip install -e .
```

本仓库当前使用 Conda 环境 `music-sep` 和 Python 3.11 进行开发：

```bash
conda activate music-sep
pip install -e ".[dev]"
```

安装完成后即可使用 `music-sep` 命令。

## 快速开始

当前 CLI 必须显式使用 `separate` 子命令。

```bash
# 基础分离（4 轨：vocals / drums / bass / other）
music-sep separate song.mp3

# 仅提取人声 + 伴奏（卡拉 OK 模式）
music-sep separate song.mp3 --two-stems vocals

# 分离 + 歌词识别
music-sep separate song.mp3 --lyrics

# 分离 + 可视化
music-sep separate song.mp3 --visualize

# 全量输出：分离 + 歌词 + 可视化
music-sep separate song.mp3 --lyrics --visualize

# 使用 6 轨模型（+ guitar, piano）
music-sep separate song.mp3 --model htdemucs_6s

# 显示 CLI 层执行计划，但不运行 Pipeline
music-sep separate song.mp3 --lyrics --visualize --dry-run

# 使用配置文件
music-sep separate song.mp3 --config music-sep.toml
```

## 命令参考

### `music-sep separate`

```text
music-sep separate <input> [OPTIONS]
```

当前版本尚未实现 `music-sep <input>` 默认命令别名。脚本和 AI 自动化应使用显式子命令。

#### 分离参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `htdemucs` | Demucs 模型名称 |
| `--device` | `auto` | 计算设备：cpu / cuda / mps / auto |
| `--shifts` | `1` | 随机偏移次数（越多质量越高，速度越慢） |
| `--overlap` | `0.25` | 偏移重叠比例（0.0–1.0） |
| `--two-stems` | — | 输出指定音轨及其残差（例如 `vocals`） |
| `--stems` | — | 输出选定音轨；多个音轨需重复传入该参数 |

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
| `--lyrics` / `--no-lyrics` | 禁用 | 启用或禁用歌词识别 |
| `--whisper-model` | `medium` | Whisper 模型大小 |
| `--whisper-device` | `auto` | 请求设备：cpu / cuda / mps / auto。CTranslate2 不在 MPS 上运行，因此 MPS 请求会回退到 CPU，并提供可见的回退原因 |
| `--language` | 自动检测 | 强制语言代码（例如 zh / en / ja） |
| `--lyrics-format` | `srt` | 歌词格式：srt / lrc / vtt / txt / json |

#### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--visualize` / `--no-visualize` | 禁用 | 启用或禁用可视化 |
| `--viz-types` | 全部 | 可视化类型；可针对 waveform / spectrogram / mel 重复传入 |

#### 输出与执行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `demo` | 输出根目录 |
| `--format` | `wav` | 输出音频格式：wav / mp3 / flac |
| `--bitrate` | `128k` | MP3 比特率（`8k`–`320k`） |
| `--overwrite` / `--no-overwrite` | 否 | 替换已有歌曲输出目录 |
| `--config` | — | TOML 配置文件 |
| `--dry-run` | 否 | 打印合并后的 CLI 计划，并在构造 Pipeline 前退出 |
| `-v`, `--verbose` | 否 | 启用调试日志 |
| `-q`, `--quiet` | 否 | 仅显示错误；同时指定时 verbose 优先 |

### `music-sep models`

列出可用的 Demucs 模型，以及基于缓存目录推测的安装状态。

```bash
music-sep models
music-sep models --installed-only
```

安装状态只是启发式判断，不代表 checkpoint 完整性或实际可用性。首次真实运行时，上游库仍可能下载模型。

### `music-sep info`

显示音频文件元信息。当前实现会使用 librosa 解码音频，并查询 SoundFile 元数据，因此编解码支持仍取决于本地运行环境。

```bash
music-sep info song.mp3
```

## 配置文件

项目支持四个 TOML 表。有效值按照 **CLI 非 `None` > TOML 非 `None` > 内置默认值** 选择，合并后再进行校验。

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

重要的合并与校验规则：

- 显式 CLI 布尔值（包括 `False`）会覆盖 TOML 值。
- `stems` 和 `types` 列表整体替换较低优先级列表，不会拼接。
- `two_stems` 与 `stems` 互斥。
- `shifts` 必须是非布尔、且 >= 1 的整数。
- `overlap` 必须是位于闭区间 0.0–1.0 的有限数。
- 未知 TOML 表和字段会产生警告并被忽略。
- 已知 TOML section 必须是 table。
- MP3 比特率必须使用 `<整数>k` 语法，并位于 `8k`–`320k`。

## 输出结构

输入为 `song.mp3`、使用默认输出根目录时：

```text
demo/
└── song/
    ├── stems/
    │   ├── vocals.wav
    │   ├── drums.wav
    │   ├── bass.wav
    │   └── other.wav
    ├── lyrics.srt                  # 仅在 --lyrics 成功时写入
    └── visualizations/             # 普通运行会准备该目录
        ├── waveform_original.png   # 仅在 --visualize 时写入 PNG
        ├── waveform_vocals.png
        ├── spectrogram_original.png
        └── ...
```

支持的输入文件扩展名为 `.mp3`、`.wav`、`.flac`、`.ogg`、`.m4a`、`.aac` 和 `.wma`。扩展名被接受并不保证本地所有 codec/运行时组合都能成功解码该文件。

默认情况下，如果 `<output-dir>/<song>/` 已存在，运行会失败。使用 `--overwrite` 前，请阅读下方 AI 指南中的覆盖风险说明。

<!-- AUTO-GENERATED: AI AGENT GUIDE START -->
## AI 编程代理运行与接入指南

本节根据 `pyproject.toml`、`src/music_sep/` 下的公开兼容模块以及当前测试套件同步维护，供需要运行、测试、扩展或嵌入本项目的 AI 编程代理使用，避免重复摸索项目的行为边界。

### 1. AI 代理操作契约

以下内容应视为当前实现契约：

1. **分离是必需阶段。** Pipeline 始终执行输入验证/准备和 Demucs 分离；歌词与可视化是可选后处理阶段。
2. **使用显式 CLI 形式。** 调用 `music-sep separate <input>`，不要假定 `music-sep <input>` 是别名。
3. **保持兼容。** 除非明确记录为缺陷修复，否则必须保持现有 CLI 参数、TOML 字段/默认值/优先级、输出路径与文件名、结果字段、异常以及公开模块导入路径稳定。
4. **面向用户的运行文本使用中文。** 按项目约定，现有 CLI 输出、诊断和日志消息均为中文。
5. **不要绕过校验。** 通过 `merge_config()` 构建配置，或向 `Pipeline` 传入 `AppConfig`/`ValidatedConfig`；Pipeline 会创建并持有自己的不可变已校验快照。
6. **普通测试不得下载真实模型。** 使用假模型、可注入 port/resolver、mock 和确定性的合成音频。
7. **不得发布或提交受限制产物。** 禁止向 Git 添加模型权重、模型缓存、真实版权音乐、人声、歌词、转录文本、凭据或用户生成媒体。

### 2. 可复现环境配置

从仓库根目录执行：

```bash
conda activate music-sep
python --version                 # 开发目标：Python 3.11；包最低要求：3.10
pip install -e ".[dev]"
pip check
music-sep --version
music-sep --help
```

控制台入口定义为：

```text
music-sep = music_sep.cli:app
```

`python -m music_sep` 会进入同一个 Typer 应用，但处理音频时仍必须显式使用 `separate` 子命令。

平台说明：

- 为目标 CPU、CUDA 或 MPS 环境安装相互兼容的 `torch`/`torchaudio` 版本。
- Demucs 固定为 `4.0.1`，通过其底层 `demucs.pretrained`、`demucs.apply` 和 `demucs.separate` API 接入。不要引入不存在的 `demucs.api` 模块。
- WAV/FLAC 写入使用 SoundFile/libsndfile；MP3 写入使用 Demucs 音频路径及其编码依赖。输入解码和 `info` 行为仍取决于本地可用 codec。
- Demucs 和 Whisper 实现会延迟加载模型对象，但导入当前 Pipeline 也会导入 visualization 模块，进而导入 librosa/matplotlib。不要声称所有重型依赖都在导入层面延迟加载。

### 3. 安全的 CLI 自动化

开发或评估变更时，应使用一次性输出目录：

```bash
workdir="$(mktemp -d)"
music-sep separate /path/to/authorized-input.wav --output-dir "$workdir" --dry-run
```

命令执行前，CLI 输入参数所指文件必须存在且可读。

具有代表性的真实命令：

```bash
# 仅执行必需的分离阶段
music-sep separate song.wav --output-dir ./tmp-output

# 选择多个音轨；重复使用 --stems
music-sep separate song.wav \
  --stems vocals --stems drums \
  --output-dir ./tmp-output

# 指定目标音轨及其残差
music-sep separate song.wav \
  --two-stems vocals \
  --output-dir ./tmp-output

# 启用可选歌词和可视化阶段
music-sep separate song.wav \
  --lyrics --lyrics-format lrc \
  --visualize --viz-types waveform --viz-types mel \
  --output-dir ./tmp-output
```

当前 `--dry-run` 的限制：

- CLI 会合并并校验静态配置、打印计划，然后在构造 `Pipeline` **之前**退出。
- 它不会解析运行时设备、加载模型、校验模型特定 stem 名称、校验 codec 可用性、通过 Pipeline 检查输出冲突、创建目录或写入文件。
- 因此，dry-run 成功不代表真实运行一定成功。

### 4. 配置与不可变边界

`music_sep.config` 中的兼容配置对象是可变 dataclass：

- `AppConfig`
- `SeparationConfig`
- `LyricsConfig`
- `VisualizationConfig`
- `OutputConfig`

领域执行边界使用不可变且带 slots 的 dataclass：

- `RawConfig`：优先级已经合并，但值尚未被信任。
- `ValidatedConfig`：规范化的逻辑配置；请求设备仍可能为 `auto`。
- `RuntimeResolution`：请求设备、实际设备、后端和可选回退原因。
- `ResolvedConfig`：已校验请求，以及某次运行中各阶段对应的运行时解析结果。

使用转换函数，不要手工复制字段：

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

两个转换方向都会分离列表值。即使调用方直接传入 `ValidatedConfig`，`Pipeline` 也会重新校验，而不是仅信任类型标注或调用方持有的状态。

### 5. Python Pipeline 接入

包根目录只导出 `__version__`；运行 API 应从对应的已记录模块导入。

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

当前结果字段为：

```text
PipelineResult(
    input_file,
    output_paths,
    stems={},
    lyrics_path=None,
    visualization_paths=None,
)
```

调用方需要了解的运行语义：

- Pipeline 在构造时复制并校验配置，不会修改调用方的 `AppConfig`。
- 每次 `run()` 都会重新解析分离设备。
- 只有必需的分离阶段成功后，才会解析歌词设备。
- 歌词阶段优先使用已生成的 `vocals` 音轨；不存在时回退使用原始输入。
- 歌词和可视化异常目前会分别记录并吞掉。因此，调用成功返回时可能只有分离音轨，而缺少部分可选产物。
- 即使歌词设备解析或转录失败，可视化阶段仍可继续运行。
- 每次运行开始时都会重置 `resolved_config`，避免暴露上一次运行的陈旧状态。启用歌词时，如果 dry-run 或运行在歌词设备成功解析前失败，由于完整快照需要歌词解析结果，`resolved_config` 可能为 `None`。

当前公开模块还提供以下底层兼容 API：

```python
from music_sep.lyrics import LyricsTranscriber, TranscriptionResult
from music_sep.outputs import OutputPaths, ensure_output_dirs, resolve_output_paths
from music_sep.separation import SeparationEngine
from music_sep.utils import detect_device, resolve_device, validate_input_file
from music_sep.visualization import Visualizer
```

`SeparationEngine` 还支持注入 `Separator` 和 `AudioWriter` 实现，`Pipeline` 支持注入 `RuntimeResolver`。测试中应优先使用这些边界，而不是修改私有字段或直接调用第三方库。

### 6. 运行时设备矩阵

设备请求与后端实际使用的设备分别表示。

| 阶段/后端 | `auto` 优先级 | 显式 MPS | 显式请求不可用的 CUDA/MPS |
|-----------|---------------|----------|-----------------------------|
| Demucs / PyTorch | CUDA → MPS → CPU | PyTorch 报告可用时使用 MPS | 抛出设备错误 |
| Whisper / CTranslate2 | CTranslate2 CUDA → CPU | 解析为 CPU，并记录/输出回退原因 | CTranslate2 无法使用 CUDA 时，CUDA 请求报错；MPS 请求回退 CPU |

重要接入规则：

- 使用 CTranslate2 自身的设备 API 探测其 CUDA 能力，不使用 `torch.cuda.is_available()` 推断。
- 本项目中的 CTranslate2 永远不会运行在 MPS 上。
- Whisper 在 CUDA 上使用 `float16`，在 CPU 上使用 `int8`。
- 可选 Whisper 设备不可用时，不得阻止必需的 Demucs 分离完成。
- 需要请求设备、实际设备和回退原因时使用 `resolve_device()`；`detect_device()` 是仅返回字符串的旧兼容 API。

无需构造模型的示例：

```python
from music_sep.utils import resolve_device

separation_runtime = resolve_device("auto", backend="torch")
lyrics_runtime = resolve_device("mps", backend="ctranslate2")

assert lyrics_runtime.actual == "cpu"
assert lyrics_runtime.fallback_reason is not None
```

### 7. 输出、副作用与覆盖安全

输出路径根据 `input_file.stem` 生成：

```text
<output_dir>/<song_name>/
├── stems/<stem>.<wav|flac|mp3>
├── lyrics.<srt|lrc|vtt|txt|json>
└── visualizations/<type>_<original|stem>.png
```

单个 stem 文件会先编码到同级临时位置，再替换到最终路径。**歌曲目录整体目前尚不具备事务性。** 当前 `overwrite=True` 会在分离开始前递归删除已有的最终歌曲目录。如果后续必需或可选操作失败，旧目录不会恢复。

在实现事务发布前，AI 代理必须遵守以下安全规则：

1. 不要在必须保留数据的输出目录上使用 `--overwrite`。
2. 每次自动运行或测试都优先使用新的临时 `--output-dir`。
3. 如果必须替换，应先把旧输出复制或备份到目标树外，并向用户报告该操作；不得静默删除。
4. 自动化代码不得在缺少同等保护时调用 `ensure_output_dirs(..., overwrite=True)`。
5. 当前尚未实现符号链接保护、并发保护、中断恢复、目录回滚和事务锁；应按未实现处理。

### 8. 验证命令

在已激活的开发环境中，从仓库根目录执行：

```bash
# 完整测试与非 slow 测试门禁
pytest tests/ -v
pytest -m "not slow"

# 聚焦测试示例
pytest tests/test_config.py -v
pytest tests/test_config.py::test_merge_config_defaults -v

# 静态质量门禁
ruff format --check src tests
ruff check src tests
mypy src

# 依赖与打包门禁
pip check
python -m build

# 基础导入与入口 smoke
python -c "import music_sep; import music_sep.cli; print(music_sep.__version__)"
music-sep --version
music-sep --help
```

测试编写约束：

- 使用固定随机种子生成短音频，或使用现有合成 fixture。
- 注入假的 separator、writer、Whisper 模型和运行能力探针。
- 需要真实模型下载或真实硬件的测试应标记为 `slow`，不得加入普通离线门禁。
- 不得根据开发机器真实的 CUDA 或 MPS 可用性决定测试通过或失败。
- 不得把版权音频或歌词作为已提交 fixture。

### 9. AI 代理变更定位表

修改对应行为前应先检查以下文件：

| 关注点 | 事实来源 |
|--------|----------|
| 包依赖、入口、测试 marker | `pyproject.toml` |
| CLI 命令、参数、dry-run、退出映射 | `src/music_sep/cli.py` |
| 默认值、TOML schema、优先级、兼容 DTO | `src/music_sep/config.py` |
| 不可变配置与运行时不变量 | `src/music_sep/domain/config.py` |
| 模型、格式、扩展名、设备目录 | `src/music_sep/domain/catalog.py` |
| 设备能力与回退规则 | `src/music_sep/adapters/torch_runtime.py` |
| Pipeline 阶段顺序与可选失败行为 | `src/music_sep/pipeline.py` |
| Demucs 兼容 facade 与注入 port | `src/music_sep/separation.py` |
| Demucs 实现 | `src/music_sep/adapters/demucs_separator.py` |
| 音频编码与单文件原子发布 | `src/music_sep/adapters/torchaudio_writer.py` |
| 歌词模型与五种输出 writer | `src/music_sep/lyrics.py` |
| 可视化行为与命名 | `src/music_sep/visualization.py` |
| 输出布局与破坏性目录覆盖 | `src/music_sep/outputs.py` |
| 输入验证与兼容设备 API | `src/music_sep/utils.py` |
| 完整重构行为/风险目录 | `TECHNICAL_HANDOFF.md` |

新增内部代码时应遵循以下架构方向：

```text
domain <- ports <- adapters
                <- compatibility facades / current Pipeline
```

第三方无关的值和不变量放入 `domain`，行为契约放入 `ports`，Demucs/Torch/编码实现放入 `adapters`。现有根模块是兼容 facade，必须保留其导入路径。

### 10. 完成检查清单

AI 代理声明变更完成前，应确认：

- [ ] 请求行为由确定性测试覆盖。
- [ ] 普通测试不会下载模型，也不要求 GPU/MPS 硬件。
- [ ] `pytest tests/ -v` 与 `pytest -m "not slow"` 通过。
- [ ] Ruff format/check 与 mypy 通过。
- [ ] 如果修改打包或公开导入，构建与导入 smoke 通过。
- [ ] CLI/TOML/输出/Python 兼容性已保持，或有意缺陷修复已记录。
- [ ] 面向用户的 CLI/日志文本保持中文。
- [ ] 未破坏性覆盖已有输出。
- [ ] 未添加模型权重、缓存、凭据、真实版权媒体、歌词或转录文本。
- [ ] 可观察行为或已记录风险发生变化时，已更新 `README.md` 与 `TECHNICAL_HANDOFF.md`。
- [ ] 未触碰无关的未跟踪文件。

<!-- AUTO-GENERATED: AI AGENT GUIDE END -->

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 排除可选 slow/模型/硬件测试
pytest -m "not slow"

# 格式、代码检查和类型检查
ruff format --check src tests
ruff check src tests
mypy src

# 在本地构建发行产物
python -m build
```

## 许可证

Apache-2.0
