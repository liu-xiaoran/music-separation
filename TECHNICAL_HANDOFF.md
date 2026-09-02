# music-sep 技术交接与等价重构规格

> 本文档用于在**完整重构本项目时不丢失现有能力、外部接口和重要边界行为**。
>
> 文档区分三类内容：
>
> 1. **当前事实**：当前源码实际实现的行为；
> 2. **兼容契约**：重构后原则上应保留的对外行为；
> 3. **已知缺陷**：当前存在但不应机械复制的错误、风险或文档偏差。

---

## 0. 文档基线

| 项目 | 值 |
|---|---|
| 项目名 | `music-sep` |
| 当前版本 | `0.1.0` |
| 基线日期 | 2026-08-29 |
| Git 分支 | `main` |
| Git 提交 | `62e2ad2b12f3f04da775785a2bda4eba1b48da76` |
| Python 要求 | `>=3.10`；项目开发环境为 Conda `music-sep` / Python 3.11 |
| 构建后端 | Hatchling |
| 许可证 | Apache-2.0 |
| CLI 入口 | `music-sep = music_sep.cli:app` |
| 模块入口 | `python -m music_sep` |
| pytest 实测基线 | 两次运行均为 67 项：64 passed、3 failed、5 warnings；耗时分别为 30.40s 和 2.90s，耗时受缓存与机器状态影响，不作为固定性能指标 |
| 当前 Ruff 结果 | `ruff check src/ tests/`：30 个问题 |
| 当前 mypy 结果 | `mypy src/`：12 个错误，涉及 4 个文件 |

> 写入本文档前，工作区已有未跟踪文件 `AGENTS.md`；本次新增 `TECHNICAL_HANDOFF.md`。两者均不属于上述基线提交，本文档未修改 `AGENTS.md`。

### 0.1 重构实施状态覆盖（2026-08-30）

本小节只记录上述审计基线之后的实施状态；第 0 节及后续“当前事实”仍是提交 `62e2ad2` 的历史快照，不应据此判断最新分支是否仍存在同一缺陷。

| 范围 | 当前状态 |
|---|---|
| 历史审计基线 | `62e2ad2b12f3f04da775785a2bda4eba1b48da76`；保留原测试、Ruff、mypy 结果作为重构前证据 |
| PR 1 | 本地提交 `3df3de8`：修复 Demucs 4.0.1 归一化/反归一化与 stem 选择正确性，引入 separator/audio-writer 边界，并恢复基础质量门禁 |
| PR 2 | 当前分支 `refactor/pr2-config-device-boundary`：引入不可变 `RawConfig` / `ValidatedConfig` / `ResolvedConfig`、结构化运行时设备解析及兼容 facade；Pipeline 不再修改调用方配置；CTranslate2 使用独立 CUDA 探测，MPS 请求可见回退 CPU；不可用的可选歌词设备不会阻止必做分离 |
| 当前自动化验证 | `pytest tests/ -v` 与 `pytest -m "not slow"` 均为 198 passed、5 warnings；`ruff check src tests`、`ruff format --check src tests`、`mypy src`、`pip check`、构建及干净 wheel 导入 smoke 均通过 |
| 仍待实施 | 输出目录事务、并发保护、overwrite 回滚与旧正式结果保护仍属于 PR 3；在 PR 3 完成前，第 14 节对应的数据安全风险仍有效 |

本文档列出的 257 个唯一测试用例 ID 是完整重构的目标验收目录，不等同于当前 pytest 收集的 198 个自动化测试项。

---

## 1. 项目定位与能力摘要

`music-sep` 是一个本地命令行音频处理工具，统一编排以下能力：

1. **音乐源分离**：使用 Demucs 将输入音频拆分为人声、鼓、贝斯、其他等 stem；
2. **歌词转写**：可选使用 faster-whisper 对优先分离出的人声轨进行转写；
3. **音频可视化**：可选使用 librosa 与 matplotlib 为原音频和所有输出 stem 生成图像；
4. **音频信息查看**：输出格式、时长、采样率、声道数和文件大小；
5. **模型目录展示**：展示项目允许使用的 Demucs 模型，并启发式标注本地安装状态；
6. **三层配置**：CLI 参数覆盖 TOML，TOML 覆盖内置默认值；
7. **统一输出目录**：按输入文件主文件名建立歌曲目录。

### 1.1 当前实现最重要的事实

- `separate` 是主处理命令，分离阶段始终执行；歌词和可视化只是可选后续阶段。
- 当前 CLI **没有**实现 README 所声称的 `music-sep song.mp3` 默认命令别名；实际必须使用 `music-sep separate song.mp3`。
- 当前分离实现使用 Demucs 4.0.x 的低层 API，而不是不存在的 `demucs.api`。
- ctranslate2/faster-whisper 不支持 MPS；歌词自动设备选择只会返回 CUDA 或 CPU。
- 歌词或可视化失败会被记录并吞掉，分离成功时 CLI 仍可整体成功退出。
- 当前分离代码存在严重的音频归一化/反归一化错误，详见第 14 节。

### 1.2 “等价重构”的定义

重构后的系统至少应满足：

- 保留当前有效的 CLI 命令、选项、配置字段、默认值、输出命名和格式；
- 对当前文档与实现冲突的地方作出明确选择，并用测试固定最终决定；
- 不把已确认的音频错误、破坏性覆盖、错误测试等缺陷当作兼容目标；
- 所有可观察变化均有迁移说明或兼容层；
- 在真实 Demucs、Whisper、音频编码器和多设备环境下完成分层验证。

---

## 2. 仓库结构与模块职责

```text
music-separation/
├── src/music_sep/
│   ├── __init__.py          # __version__ = "0.1.0"
│   ├── __main__.py          # python -m music_sep 入口
│   ├── cli.py               # Typer CLI：separate / models / info / --version
│   ├── config.py            # 配置 dataclass、TOML 加载、三层合并、校验
│   ├── constants.py         # 输入格式、模型列表、可视化类型、默认采样率常量
│   ├── exceptions.py        # 项目异常层次
│   ├── outputs.py           # 输出路径解析、目录创建、文件命名
│   ├── pipeline.py          # 四阶段编排与降级策略
│   ├── separation.py        # Demucs 低层 API、stem 选择与音频写出
│   ├── lyrics.py            # faster-whisper 封装与歌词格式写出
│   ├── visualization.py     # 波形、STFT、Mel 图像生成
│   └── utils.py             # 日志、输入校验、设备选择、时长格式化
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_lyrics.py
│   ├── test_outputs.py
│   ├── test_pipeline.py
│   ├── test_separation.py
│   ├── test_utils.py
│   └── test_visualization.py
├── README.md                # 英文主文档
├── README_CN.md             # 中文文档
├── pyproject.toml
├── LICENSE
├── CLAUDE.md
└── .gitignore
```

### 2.1 当前缺失的工程化资产

当前没有发现：锁文件、Conda 环境声明文件、Dockerfile、CI 工作流、发布流水线、覆盖率阈值、`[tool.mypy]` 配置、真实模型/编解码器集成测试基线，以及性能、显存、内存或输出质量基线。

---

## 3. 架构与调用链

### 3.1 逻辑分层

```text
用户 / Shell
    │
    ▼
Typer CLI (`cli.py`)
    │ 解析参数、配置日志、合并配置、打印计划、映射退出码
    ▼
配置层 (`config.py`)
    │ CLI > TOML > 默认值；生成 AppConfig
    ▼
Pipeline (`pipeline.py`)
    ├── Stage 1：输入校验、设备解析、输出路径与目录
    ├── Stage 2：Demucs 分离（必做）
    ├── Stage 3：Whisper 歌词识别（可选、失败降级）
    └── Stage 4：原音频与所有 stem 可视化（可选、失败降级）
    │
    ▼
输出文件系统 (`outputs.py`)
```

### 3.2 实际调用顺序

1. `music_sep.cli:separate()` 接收 Typer 参数；
2. `setup_logging(verbose, quiet)` 配置 `music_sep` logger；
3. `merge_config(...)` 可选读取 TOML、逐字段执行 CLI > TOML > 默认值、构造并校验 `AppConfig`；
4. CLI 向 stdout 打印一次执行计划；
5. CLI `--dry-run` 在此处直接退出，**不会调用 Pipeline**；
6. 非 dry-run 创建 `Pipeline(app_config)`；
7. `Pipeline.run()` 依次调用输入校验、设备解析、输出准备、Demucs、可选 Whisper、可选可视化；
8. CLI 将 `OutputExistsError` 或其他异常映射为退出码 1。

### 3.3 数据流

```text
输入音频
  ├─> Demucs load_track
  │      └─> apply_model
  │             └─> stem tensor
  │                    └─> WAV / FLAC / MP3 文件
  │
  ├─> 歌词源选择
  │      ├─ 若存在 vocals：使用 vocals 文件
  │      └─ 否则：使用原输入音频
  │             └─> Whisper segments
  │                    └─> SRT/LRC/VTT/TXT/JSON
  │
  └─> 可视化
         ├─ 原输入音频
         └─ 每个已输出 stem
                └─> waveform/spectrogram/mel PNG
```

### 3.4 Pipeline 的失败语义

| 阶段 | 是否必做 | 当前失败行为 | 当前 CLI 结果 |
|---|---:|---|---|
| 配置合并/校验 | 是 | 异常被 CLI 捕获，打印“配置错误” | exit 1 |
| 输入/设备/路径准备 | 是 | 异常传播给 CLI | exit 1 |
| Demucs 分离 | 是 | 通常包装为 `AudioProcessingError` 后传播 | exit 1，可能留下部分输出 |
| 歌词识别 | 否 | 捕获任意 `Exception`，日志记录并继续 | 通常 exit 0 |
| 可视化 | 否 | 捕获任意 `Exception`，日志记录并继续 | 通常 exit 0 |
| 单种可视化 | 否 | `Visualizer.generate()` 内部吞掉并继续其他类型 | 通常 exit 0 |

> 重构时必须决定“可选阶段失败仍返回成功”是否继续作为兼容契约。建议保留默认兼容行为，但在结构化结果、摘要和可选 `--strict` 模式中显式暴露部分失败。

---

## 4. 打包、安装与运行环境

### 4.1 构建与入口

- `pyproject.toml:1-3` 使用 Hatchling；
- `pyproject.toml:5-22` 定义项目元数据；
- `pyproject.toml:33-34` 注册 `music-sep = "music_sep.cli:app"`；
- `src/music_sep/__main__.py:1-3` 支持 `python -m music_sep`。

### 4.2 声明的运行依赖

| 依赖 | 版本约束 | 用途 | 备注 |
|---|---|---|---|
| `typer` | `>=0.12,<1.0` | CLI | Rich 帮助输出 |
| `demucs` | `>=4.0` | 音乐分离 | 源码针对 4.0.x 低层 API；当前没有上界 |
| `faster-whisper` | `>=1.0` | 歌词识别 | 后端 ctranslate2 不支持 MPS |
| `librosa` | `>=0.10` | 音频读取、时长、频谱 | `info` 和可视化都会全量读取 |
| `soundfile` | `>=0.12` | 测试音频生成、声道信息 | `info` 直接依赖 |
| `matplotlib` | `>=3.7` | PNG 图像 | 使用 Agg 后端 |
| `numpy` | `>=1.24` | 数值处理和测试音频 | 直接依赖 |
| `torch` | `>=2.0` | Demucs、设备检测 | GPU/MPS 行为依赖安装版本 |
| `pydub` | `>=0.25` | 当前源码未使用 | 建议移除或补充明确用途 |
| `tomli` | Python `<3.11` | TOML 解析兼容 | 3.11+ 使用 `tomllib` |

### 4.3 实际需要但元数据未直接声明的依赖

- `torchaudio`：`src/music_sep/separation.py:166-203` 在所有音频写出路径入口处直接导入；
- `torchcodec`：项目说明指出较新 torchaudio 保存音频可能要求它；
- `ffmpeg`：README 要求系统级安装，MP3/部分输入编解码通常依赖它。

> `_save_audio()` 在判断输出格式之前先 `import torchaudio`，因此即使 MP3 本可走 Demucs 的 `save_audio()`，缺少 torchaudio 仍会先失败。

### 4.4 开发依赖和命令

```bash
conda activate music-sep
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
mypy src/
```

当前开发依赖：pytest、pytest-cov、pytest-mock、ruff、mypy。

### 4.5 建议的依赖治理

1. 明确支持的 Demucs 精确版本或兼容区间，至少针对 `4.0.1` 建立测试；
2. 把 `torchaudio` 作为直接依赖或可选 codec extra；
3. 根据实际 torchaudio 版本声明 `torchcodec`；
4. 把 ffmpeg 变成启动前检查项，并给出中文诊断；
5. 移除未使用的 pydub，或把 MP3 编码统一封装到它；
6. 增加锁文件/Conda 环境文件；
7. 分离 CPU、CUDA、MPS 的可安装依赖说明；
8. 添加 CI 中可运行的最小 CPU 依赖组合。

---

## 5. CLI 对外契约

主要实现：`src/music_sep/cli.py`。

### 5.1 顶层命令

| 命令 | 当前能力 | 退出语义 |
|---|---|---|
| `music-sep --help` | 显示顶层帮助和子命令 | 0 |
| `music-sep --version` | 输出 `music-sep 0.1.0` | 0 |
| `music-sep separate <input>` | 执行完整处理流水线 | 成功 0；配置/处理失败 1 |
| `music-sep models` | 列出允许的 Demucs 模型 | 通常 0 |
| `music-sep info <input>` | 显示音频元信息 | 成功 0；Typer/解码异常非 0 |

### 5.2 README 声称但当前不存在的默认命令

README 声称 `music-sep song.mp3` 与 `music-sep separate song.mp3` 等价，但当前 `app.callback()` 不接收输入参数，也不转发到 `separate()`。当前真实可用形式只有：

```bash
music-sep separate song.mp3
```

重构时必须二选一：推荐实现兼容默认命令并保留显式 `separate`，或者删除所有默认命令文档和示例。

### 5.3 `separate` 参数全集

| CLI 参数 | 配置字段 | 类型 | 内置默认 | 当前约束/语义 |
|---|---|---|---|---|
| `<input>` | — | `Path` | 必填 | Typer 要求存在、可读；Pipeline 再要求普通文件且扩展名受支持 |
| `--model` | `separation.model` | str | `htdemucs` | 必须在固定模型表中 |
| `--device` | `separation.device` | str | `auto` | 运行时支持 cpu/cuda/mps/auto |
| `--shifts` | `separation.shifts` | int | `1` | 校验 `>=1` |
| `--overlap` | `separation.overlap` | float | `0.25` | 校验 `[0.0, 1.0]` |
| `--two-stems` | `separation.two_stems` | str/None | None | 输出目标 stem 和 `no_<stem>`；未提前校验 stem 名称 |
| `--stems` | `separation.stems` | list[str]/None | None | 可重复；仅保存匹配项；可能过滤成空集合 |
| `--lyrics/--no-lyrics` | `lyrics.enabled` | Optional[bool] | False | 三态参数允许 CLI 显式覆盖 TOML |
| `--whisper-model` | `lyrics.whisper_model` | str | `medium` | 固定白名单 |
| `--whisper-device` | `lyrics.whisper_device` | str | `auto` | ctranslate2 只支持 cpu/cuda/auto；mps 显式报错 |
| `--language` | `lyrics.language` | str/None | 自动检测 | 未验证语言代码 |
| `--lyrics-format` | `lyrics.output_format` | str | `srt` | srt/lrc/vtt/txt/json |
| `--visualize/--no-visualize` | `visualization.enabled` | Optional[bool] | False | 三态参数允许 CLI 显式覆盖 TOML |
| `--viz-types` | `visualization.types` | list[str] | waveform/spectrogram/mel | 可重复；列表整体覆盖 TOML |
| `--output-dir` | `output.output_dir` | Path | `demo` | 相对路径相对于当前工作目录 |
| `--format` | `output.format` | str | `wav` | wav/mp3/flac |
| `--bitrate` | `output.bitrate` | str | `128k` | 只用于 MP3；当前无格式校验 |
| `--overwrite/--no-overwrite` | `output.overwrite` | Optional[bool] | False | True 时递归删除整首歌输出目录 |
| `-v/--verbose` | — | bool | False | logger DEBUG；与 quiet 同时出现时 verbose 优先 |
| `-q/--quiet` | — | bool | False | logger ERROR；不抑制 stdout 计划、摘要和 Demucs progress |
| `--config` | — | Path/None | None | TOML 配置文件路径 |
| `--dry-run` | — | bool | False | CLI 仅打印配置计划后退出，不调用 Pipeline |

### 5.4 `models`

参数 `--installed-only` 只展示启发式判断为已安装的模型。

| 模型 | 描述 |
|---|---|
| `htdemucs` | Hybrid Transformer 4 轨：drums、bass、other、vocals |
| `htdemucs_6s` | Hybrid Transformer 6 轨：额外 guitar、piano |
| `htdemucs_ft` | 微调版 Hybrid Transformer 4 轨 |
| `mdx` | MDX 挑战基线 4 轨 |
| `mdx_extra` | MDX 增强 4 轨 |
| `hdemucs_mmi` | Hybrid Demucs 2 轨：vocals + accompaniment |

当前安装检测只扫描 `~/.cache/torch/hub/` 下目录名包含 `demucs` 且包含模型名的目录，不检查真正 checkpoint 文件，因此结果不可靠。

### 5.5 `info`

1. `librosa.load(..., sr=None)` 全量加载音频，默认转单声道；
2. `librosa.get_duration()` 计算时长；
3. `soundfile.info()` 获取声道数；
4. `Path.stat()` 获取文件大小；
5. 打印文件名、格式、`MM:SS` 时长、采样率、声道数和 MiB 大小。

当前没有统一异常包装；长音频会全量加载，某些 librosa 可解码但 libsndfile 不支持的格式可能在 `soundfile.info()` 失败。

---

## 6. 配置系统

实现：`src/music_sep/config.py`。

### 6.1 数据模型与默认值

```python
AppConfig(
    separation=SeparationConfig(
        model="htdemucs", device="auto", shifts=1, overlap=0.25,
        two_stems=None, stems=None,
    ),
    lyrics=LyricsConfig(
        enabled=False, whisper_model="medium", whisper_device="auto",
        language=None, output_format="srt",
    ),
    visualization=VisualizationConfig(
        enabled=False, types=["waveform", "spectrogram", "mel"],
    ),
    output=OutputConfig(
        output_dir=Path("demo"), format="wav", bitrate="128k", overwrite=False,
    ),
)
```

### 6.2 TOML Schema

```toml
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
output_format = "srt"

[visualization]
enabled = false
types = ["waveform", "spectrogram", "mel"]

[output]
output_dir = "demo"
format = "wav"
bitrate = "128k"
overwrite = false
```

### 6.3 合并规则

逐字段执行 `CLI 非 None 值 > TOML 非 None 值 > 内置默认值`。

关键细节：

- `--no-lyrics`、`--no-visualize`、`--no-overwrite` 的 False 是显式值，可覆盖 TOML True；
- list 字段整体替换，不做合并；
- 只读取四个已知 section；未知 section 和未知字段静默忽略；
- section 如果不是 TOML table，会抛 `ConfigurationError`；
- `output.output_dir` 必须是字符串或 Path；
- TOML 读取失败统一包装为 `ConfigurationError`。

### 6.4 当前已校验的字段

- Demucs/Whisper 模型白名单；
- 歌词格式 srt/lrc/vtt/txt/json；
- 输出音频格式 wav/mp3/flac；
- 可视化类型 waveform/spectrogram/mel；
- `two_stems` 与 `stems` 互斥；
- 两个 list 字段的容器类型；
- overlap 范围；
- shifts 下界。

> **调用边界限制：**上述校验只在 `merge_config()` 返回前自动执行。直接构造 `AppConfig(...)` 不会触发 `validate_config()`，`Pipeline.__init__()` 和 `Pipeline.run()` 也不会再次校验。因此，当前 CLI 路径通常经过校验，但直接 Python API 可以把非法配置传入 Pipeline。重构后应让应用边界只接受 `ValidatedConfig`，或在 Pipeline 入口显式校验。

### 6.5 当前未充分校验的字段

- separation/whisper 设备字符串；
- bitrate 形式和范围；
- TOML bool 字段真实类型；
- shifts 必须为整数；
- overlap 必须为数值且有限；
- `two_stems`/`stems` 是否属于实际模型 sources；
- stems 元素类型和空输出；
- language 代码；
- 未知字段；
- 白名单与实际 Demucs 版本的一致性。

---

## 7. 输入、设备与日志

### 7.1 输入格式

`separate`/Pipeline 的输入校验当前接受 `.mp3 .wav .flac .ogg .m4a .aac .wma`。路径先绝对化，必须存在并为普通文件；扩展名大小写不敏感；只验证后缀，不验证内容和实际 codec。README 未完整列出 ogg/m4a/aac/wma。

`info` 命令不调用 `validate_input_file()`，Typer 只要求路径存在且可读，之后直接交给 librosa 和 soundfile 解码。因此，7 种扩展白名单不是 `info` 的入口限制；`info` 实际可用格式取决于两套解码库及本机 codec。

### 7.2 设备选择矩阵

| backend | preference | 当前结果 |
|---|---|---|
| torch | cpu | cpu |
| torch | cuda | 可用则 cuda，否则错误 |
| torch | mps | 可用则 mps，否则错误 |
| torch | auto | cuda > mps > cpu |
| ctranslate2 | cpu | cpu |
| ctranslate2 | cuda | 可用则 cuda，否则错误 |
| ctranslate2 | mps | 始终错误 |
| ctranslate2 | auto | cuda > cpu，不选择 mps |
| 任意 | 未知 preference | `DeviceNotAvailableError` |
| 未知 backend | auto | `DeviceNotAvailableError` |
| 未知 backend | cpu/cuda/mps | 当前不会校验 backend，而是直接执行对应 preference 分支；这是需要测试并在重构中明确的现有边界行为 |

Pipeline 会把 `auto` 原地改成实际设备，因而同一个 `AppConfig` 在执行前后内容不同。

### 7.3 日志

- logger 名 `music_sep`；默认 INFO，verbose DEBUG，quiet ERROR；
- verbose 与 quiet 同时启用时 verbose 优先；
- handler 输出 stderr，格式 `HH:MM:SS [LEVEL] message`；
- 重复调用不重复添加 handler，但更新级别；
- 未设置 `propagate=False`，嵌入宿主可能重复输出；
- quiet 不抑制 stdout 计划/摘要，也不一定抑制 Demucs progress。

---

## 8. 输出路径与覆盖语义

### 8.1 目录结构

```text
<output_root>/
└── <input_file.stem>/
    ├── stems/
    │   ├── <stem>.<wav|flac|mp3>
    │   └── ...
    ├── lyrics.<srt|lrc|vtt|txt|json>
    └── visualizations/
        ├── waveform_original.png
        ├── waveform_<stem>.png
        ├── spectrogram_original.png
        ├── spectrogram_<stem>.png
        ├── mel_original.png
        └── mel_<stem>.png
```

### 8.2 关键语义

- 歌曲目录只使用 `input_file.stem`，因此 `song.mp3` 和 `song.flac` 冲突；
- base 已存在且 overwrite=False 时直接拒绝；
- overwrite=True 时递归删除整首歌目录再创建；
- 输出非事务；失败可导致旧结果丢失、新结果残缺；
- stem 逐个写出，后续失败时前面文件保留；
- 即使未启用可视化也创建 `visualizations/`；
- 歌词和可视化 API 不创建父目录；
- resolve 与 mkdir 分离，存在竞态窗口。

### 8.3 推荐事务方案

```text
<song>.tmp-<uuid>/  --全部阶段完成-->  原子 rename 到 <song>/
```

覆盖时先完整写 staging，成功后备份旧目录并原子替换；失败时回滚，不能先删除旧正式产物。

---

## 9. Demucs 分离能力

### 9.1 模型加载

`SeparationEngine` 通过 `_load_model()` 惰性导入 `get_model` 和 `BagOfModels`，加载模型、移动到设备，并记录模型名、采样率、sources。导入/get_model 异常包装为 `ModelNotFoundError`；`model.to()` 异常当前未包装。

### 9.2 推理流程

使用 Demucs 4.0.x 低层 API：

```python
from demucs.pretrained import get_model
from demucs.separate import load_track
from demucs.apply import apply_model
```

流程为 `load_track` → 添加 batch 维并移动设备 → `apply_model(shifts, overlap, progress=True, device)` → 去 batch 维 → stem 选择 → 逐轨写出。CUDA OOM 转为带中文建议的 `AudioProcessingError`。

### 9.3 标准、two-stems 与过滤

- 标准模式按 `model.sources` 输出；
- `two_stems="vocals"` 当前输出 `vocals` 和 `no_vocals`；残差为其他 sources 求和；
- two_stems 未提前验证，`sources.index()` 可在完成推理后抛 ValueError；
- stems 过滤在完整推理后执行，不节省推理成本；未知 stem 静默忽略，全部不匹配可成功返回空字典；
- 对 BagOfModels 动态设置 `model.two_stems` 在 Demucs 4.0.1 中未必被消费。

### 9.4 输出编码

| 格式 | 当前实现 |
|---|---|
| WAV | `torchaudio.save`，PCM signed 16-bit |
| FLAC | `torchaudio.save(format="flac")` |
| MP3 | 优先 Demucs `save_audio`；仅 ImportError 时回退 torchaudio |

1D tensor 补 channel 维；大于 2D 时直接 squeeze。bitrate 仅 `rstrip("kK")` 后转 int。文件非原子写入。

---

## 10. Whisper 歌词能力

### 10.1 模型和转写

`LyricsTranscriber` 惰性创建 faster-whisper：CUDA 使用 float16，CPU 使用 int8，MPS 不支持。固定调用参数为 `beam_size=5`、`vad_filter=True` 和可选 language。segment 保存 start/end/strip 后的 text，并记录识别语言和置信度。

### 10.2 音频源

有 vocals stem 时优先使用，否则回退原输入。若 stems 过滤未包含 vocals，也会回退原输入。

### 10.3 输出格式

| 格式 | 结构 | 时间戳 |
|---|---|---|
| SRT | 序号、起止时间、文本 | `HH:MM:SS,mmm` |
| LRC | 每行起始时间和文本 | `[mm:ss.xx]` |
| VTT | WEBVTT、起止时间、文本 | `HH:MM:SS.mmm` |
| TXT | 每段一行 | 无 |
| JSON | segment 数组 | 数值秒 |

JSON 当前不保存语言元数据。负时间 clamp 到 0；毫秒/厘秒截断而非四舍五入；不验证 end>=start、NaN/Inf。写文件只包装 OSError，坏 segment 可能泄漏 KeyError/TypeError。Pipeline 会吞掉全部歌词异常。

---

## 11. 可视化能力

- import 时使用 matplotlib Agg；
- librosa 全量读取且默认 mono；
- 12×4 inch，150 DPI，PNG；
- waveform 使用 waveshow；spectrogram 使用 STFT/log 频率轴；mel 使用 Mel 频谱；
- 对原音频和每个实际输出 stem 生成；
- 理论数量为 `(1 + stem 数) × 类型数`；
- 单种图失败继续，音频加载失败抛 VisualizationError 后由 Pipeline 吞掉；
- figure 在 finally 中关闭，但缺少资源泄漏回归；
- 长音频和多 stem 重复全量加载，内存风险高。

---

## 12. 结果对象与异常体系

### 12.1 数据对象

```python
OutputPaths(base_dir, stems_dir, lyrics_path, viz_dir)
PipelineResult(input_file, output_paths, stems={}, lyrics_path=None, visualization_paths=None)
TranscriptionResult(segments, language, language_probability)
```

### 12.2 异常层次

```text
MusicSepError
├── UnsupportedFormatError
├── DeviceNotAvailableError
├── ModelNotFoundError
├── AudioProcessingError
├── TranscriptionError
├── VisualizationError
└── ConfigurationError

OutputExistsError(MusicSepError)  # 定义在 outputs.py
```

重构时若外部调用方直接 import，应保留导入路径或提供 re-export 兼容层。

---

## 13. 必须保留的兼容契约

### CLI

- `music-sep` 命令名；`separate/models/info`；`--version`；全部现有选项；正反 bool flag；重复 list option；`python -m music_sep`；成功 0、必做阶段失败 1。

### 配置

- 四个 TOML section 和现有字段；CLI > TOML > 默认；所有默认值；list 整体替换；显式 False 覆盖 TOML True。

### 音频

- `separate`/Pipeline 的 7 种输入扩展；wav/flac/mp3 输出；6 个模型名；合法 two-stems 的输出命名；合法 stems 选择的列表语义；MP3 bitrate 参数名、默认值和合法值语义；WAV 16-bit PCM，除非提供迁移说明。未知/空 stem 结果和非法 bitrate 的当前行为属于缺陷，不是兼容目标。

### 歌词

- SRT/LRC/VTT/TXT/JSON；优先 vocals；Whisper 参数；MPS 限制；默认 medium/auto/srt。

### 可视化

- waveform/spectrogram/mel；原音频+每 stem；文件命名；单图失败继续；headless 运行。

### 输出

- `<root>/<input.stem>/...`；stems、lyrics、visualizations 路径；默认拒绝覆盖；结果对象表达产物。

---

## 14. 已知缺陷、风险与文档偏差

| 风险 ID | 级别 | 问题 | 当前行为 | 重构要求 |
|---|---|---|---|---|
| RISK-AUD-001 | P0 | Demucs 归一化/反归一化错误 | 把逐样本声道均值波形加到每个 stem，却未按官方流程标准化 | 参照锁定版本官方实现修复；加数值/重混/静音回归 |
| RISK-TST-001 | P0 | separation 测试与实现脱节 | 测试引用 `_separator/_get_separator`；实现为 `_model/_load_model` | 整体重写分离测试 |
| RISK-DOC-001 | P1 | README 默认命令不存在 | 文档用 `music-sep song.mp3`，CLI 必须有子命令 | 推荐实现默认别名并测试等价 |
| RISK-DOC-002 | P1 | “三大功能可单独使用”不真实 | Pipeline 总会先分离 | 修正文档或新增独立模式 |
| RISK-DEP-001 | P1 | torchaudio 未直接声明 | `_save_audio()` 直接 import | 明确依赖/codec extra |
| RISK-DEP-002 | P1 | torchcodec/ffmpeg 契约不完整 | 部分 codec 运行时失败 | 启动检查、矩阵和中文诊断 |
| RISK-OUT-001 | P1 | overwrite 破坏性且非事务 | 先删旧目录再处理 | staging + 原子替换 + 回滚 |
| RISK-OUT-002 | P1 | 逐 stem 非原子写出 | 后续失败留部分文件 | 临时文件后 rename |
| RISK-SEP-001 | P1 | two_stems 未提前验证 | 推理后 ValueError | 模型加载后、推理前校验 |
| RISK-SEP-002 | P1 | stems 可过滤成空仍成功 | 未知项静默忽略 | 默认报错或明确 warning |
| RISK-SEP-003 | P1 | MP3 也先强制 import torchaudio | 无关依赖阻断 MP3 writer | 延迟到对应分支 |
| RISK-SEP-004 | P1 | bitrate 无校验 | 保存阶段 ValueError | Schema 化和范围验证 |
| RISK-SEP-005 | P1 | BagOfModels 动态属性可能无效 | Demucs 4.0.1 未必消费 | 使用官方路径或后处理 |
| RISK-PIPE-001 | P1 | 可选阶段失败被完全吞掉 | exit 0，结果只能靠 None 猜测 | StageResult + 可选 strict |
| RISK-PIPE-002 | P1 | 原地修改配置 | auto 被实际设备替换 | 独立 ResolvedConfig |
| RISK-CLI-001 | P1 | 两种 dry-run 语义不一致 | CLI 不做格式/设备/输出冲突检查 | 明确 planning/validate-only |
| RISK-CLI-002 | P1 | models 检测不可靠 | 扫目录名而非 checkpoint | 使用真实缓存规则 |
| RISK-CLI-003 | P1 | info 全量加载且未包装异常 | 大文件内存高、两套 decoder 支持不一致 | 元数据/流式读取、统一错误 |
| RISK-CFG-001 | P1 | 配置类型校验不足 | 坏类型泄漏 TypeError | 严格 schema，统一 ConfigurationError |
| RISK-CFG-002 | P2 | 未知键静默忽略 | 拼写错误无提示 | warning 或 strict error |
| RISK-VIZ-001 | P2 | 未启用仍建目录 | 总建 visualizations | 决定并测试最终契约 |
| RISK-VIZ-002 | P2 | 重复全量加载 | 长音频内存高 | 缓存、降采样或分块 |
| RISK-LYR-001 | P2 | JSON 丢语言元数据 | 只写 segments | 旧格式兼容或版本化 schema |
| RISK-LYR-002 | P2 | writer 只包装 OSError | KeyError/TypeError 泄漏 | segment schema + 统一异常 |
| RISK-LOG-001 | P2 | quiet 不真正静默 | stdout/progress 仍输出 | 明确 stdout/stderr/progress |
| RISK-LOG-002 | P2 | logger 可能传播 root | 宿主可能重复日志 | 显式 propagate 策略 |
| RISK-DOC-003 | P2 | 概览漏 LRC | 参数表有 LRC | 同步文档 |
| RISK-DOC-004 | P2 | README 未列完整输入格式 | 代码支持额外 4 种 | 补格式和 codec 条件 |
| RISK-DEP-003 | P2 | pydub 未使用 | 多余依赖 | 移除或统一 codec |
| RISK-ENG-001 | P2 | 无 CI/锁/覆盖率 | 无法持续复现 | 重构前建立 |
| RISK-ENG-002 | P2 | DEFAULT_SAMPLE_RATE 未使用 | 实际跟随模型 samplerate | 删除或明确语义 |

### 14.1 P0 音频错误说明

当前概念上执行：

```python
ref = wav.mean(0)               # 随时间变化的波形
estimated = apply_model(wav)
stem = estimated[i] + ref       # 每个 stem 都加同一条波形
```

这不是 Demucs 官方反归一化，会污染每个 stem、改变振幅、降低隔离度并破坏重混关系。修复必须以锁定 Demucs 4.0.1 上游源码为准，建立零均值、DC offset、静音、低方差、可控模型输出、重混和官方 CLI 对比测试。

---

## 15. 当前质量基线

### 15.1 pytest

```bash
conda run -n music-sep pytest tests -v
```

两次独立运行结果均为 67 collected、64 passed、3 failed、5 warnings；耗时分别为 30.40s 和 2.90s。失败全部位于 `tests/test_separation.py`，原因是旧内部 API 漂移。运行耗时受模型/库缓存、文件系统和机器负载影响，本数据只证明功能结果可重复，不作为性能基线。

### 15.2 Ruff

```bash
conda run -n music-sep ruff check src/ tests/
```

30 个问题：未使用 import、无占位符 f-string、未使用变量等。29 个可普通自动修复，1 个需 unsafe fix。

### 15.3 mypy

```bash
conda run -n music-sep mypy src/
```

12 个错误，涉及 visualization、lyrics、separation、cli：librosa samplerate 类型、第三方 stub、Optional model 未收窄、get_model_info 可能 None 等。

### 15.4 fixture 风险

`tests/conftest.py` 条件导入 soundfile，但 `sample_audio` 无条件调用 `sf.write()`；缺依赖时会 NameError。应使用 `pytest.importorskip` 或显式 skip。

---

## 16. 现有测试清单

### 16.1 `test_config.py`：16 项，当前通过

默认配置；CLI>默认；TOML>默认；CLI>TOML；bool 三态；CLI False 覆盖 TOML True；非法 Demucs/Whisper 模型；非法歌词/音频/可视化格式；two_stems/stems 互斥；overlap；shifts；配置不存在；坏 TOML。

### 16.2 `test_utils.py`：15 项，当前通过

日志默认/verbose/quiet/优先级/幂等；合法 WAV/不存在/非法扩展；CPU/auto/ctranslate2 auto/MPS 错误；三种时长格式。

### 16.3 `test_outputs.py`：9 项，当前通过

基本路径、自定义歌词扩展、已存在拒绝、overwrite、创建目录、删除旧文件、stem 文件名、两种图名。

### 16.4 `test_cli.py`：9 项，当前通过

顶层帮助、版本、separate 帮助、models、info 不存在、separate 不存在、dry-run、dry-run+可选阶段、非法模型。

### 16.5 `test_lyrics.py`：8 项，当前通过

结果字段、惰性初始化、SRT/VTT/TXT/JSON、非法格式、中文 UTF-8。缺少 LRC 和 Whisper transcribe 测试。

### 16.6 `test_visualization.py`：4 项，当前通过

三种图、原音频命名、单类型、不存在音频。未验证尺寸、可读性、figure 关闭和局部失败。

### 16.7 `test_pipeline.py`：3 项，当前通过

Pipeline dry-run、不存在输入、Mock 分离阶段。未覆盖歌词、可视化、部分失败、路径冲突和配置副作用。

### 16.8 `test_separation.py`：3 项，当前全部失败

仍针对旧 Separator API，必须整体重写，不能只改属性名。

---

## 17. 完整目标测试用例矩阵

约定：P0 为重构前/合并阻断，P1 为核心完成前，P2 为发布前，P3 为长期；`slow` 表示真实模型、codec、GPU 或长耗时。

### 17.1 配置测试

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| CFG-001 | P0 | 现有 | 构造 `AppConfig()` | 默认值完整一致 |
| CFG-002 | P0 | 现有 | CLI 指定 model/device/shifts | 覆盖默认，其余不变 |
| CFG-003 | P0 | 现有 | 完整 TOML | TOML 覆盖默认 |
| CFG-004 | P0 | 现有 | TOML + 冲突 CLI | CLI 覆盖 TOML |
| CFG-005 | P0 | 现有 | bool None/True/False | 三态正确 |
| CFG-006 | P0 | 现有 | TOML True + CLI False | False 显式覆盖 |
| CFG-007 | P0 | 新增 | CLI list + TOML list | CLI 整体替换，不拼接 |
| CFG-008 | P0 | 新增 | TOML 只给部分字段 | 其余回落默认 |
| CFG-009 | P0 | 新增 | 四个 section 分别非 table | ConfigurationError 指出 section |
| CFG-010 | P0 | 新增 | output_dir 整数/数组/bool | ConfigurationError |
| CFG-011 | P0 | 现有 | 配置不存在 | ConfigurationError 含路径 |
| CFG-012 | P0 | 现有 | TOML 语法错误 | ConfigurationError 保留原因 |
| CFG-013 | P0 | 现有 | 未知 Demucs 模型 | 拒绝并列可用模型 |
| CFG-014 | P0 | 现有 | 未知 Whisper 模型 | 拒绝 |
| CFG-015 | P0 | 现有 | 非法歌词格式 | 拒绝 |
| CFG-016 | P0 | 现有 | 非法音频格式 | 拒绝 |
| CFG-017 | P0 | 现有 | 非法可视化类型 | 拒绝 |
| CFG-018 | P0 | 现有 | two_stems + stems | 拒绝 |
| CFG-019 | P0 | 部分现有 | 当前只测 overlap=1.5；新增小于 0 的边界 | 两侧越界均拒绝 |
| CFG-020 | P0 | 部分现有 | 当前只测 shifts=0；新增负数 | 0 和负数均拒绝 |
| CFG-021 | P1 | 新增 | shifts float/string/bool | 统一 ConfigurationError |
| CFG-022 | P1 | 新增 | overlap string/NaN/Inf | 统一 ConfigurationError |
| CFG-023 | P1 | 新增 | stems 非 list/含非字符串 | 拒绝并指出元素 |
| CFG-024 | P1 | 新增 | viz types 非 list/含非字符串 | 拒绝 |
| CFG-025 | P1 | 新增 | 合法/非法 bitrate | 合法规范化，非法拒绝 |
| CFG-026 | P1 | 新增 | 合法/非法设备枚举 | 统一错误语义 |
| CFG-027 | P1 | 新增 | 全部合法模型参数化 | 全部通过 |
| CFG-028 | P1 | 新增 | 全部 Whisper 模型参数化 | 全部通过 |
| CFG-029 | P1 | 新增 | 全部合法格式参数化 | 全部通过 |
| CFG-030 | P2 | 新增 | 未知 section/字段 | warning 或 strict error |
| CFG-031 | P2 | 新增 | bool 写字符串/整数 | 不允许隐式 truthy |
| CFG-032 | P2 | 新增 | 相对/绝对/~ 路径 | 规范化策略固定 |
| CFG-033 | P2 | 新增 | Python 3.10 tomli 与 3.11+ tomllib | 行为等价 |
| CFG-034 | P0 | 新增 | 直接构造非法 `AppConfig` 并传入 Pipeline，绕过 `merge_config()` | Pipeline 入口重新校验，或类型上只接受 `ValidatedConfig`，不得让非法配置进入执行阶段 |

### 17.2 日志、输入与设备

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| UTL-001 | P0 | 现有 | 默认日志 | INFO |
| UTL-002 | P0 | 现有 | verbose | DEBUG |
| UTL-003 | P0 | 现有 | quiet | ERROR |
| UTL-004 | P0 | 现有 | verbose+quiet | DEBUG |
| UTL-005 | P0 | 部分现有 | 当前只断言两次返回同一 logger；新增断言 handler 数不增长 | 无重复 handler |
| UTL-006 | P1 | 新增 | 已有 root logger | 无重复输出 |
| UTL-007 | P1 | 新增 | 捕获 stdout/stderr | 协议固定 |
| UTL-008 | P0 | 现有 | 合法 WAV（当前 fixture 已是绝对 tmp_path） | 返回绝对路径 |
| UTL-009 | P0 | 现有 | 不存在路径 | FileNotFoundError |
| UTL-010 | P0 | 新增 | 目录路径 | 明确不是文件 |
| UTL-011 | P0 | 现有 | txt 文件 | UnsupportedFormatError |
| UTL-012 | P0 | 新增 | 7 种扩展含大写 | 全部通过后缀校验 |
| UTL-013 | P1 | 新增 | 合法后缀损坏内容 | 解码层明确错误 |
| UTL-014 | P1 | 新增 | Unicode/空格/长文件名 | 正确解析 |
| DEV-001 | P0 | 现有 | cpu | cpu |
| DEV-002 | P0 | 新增 | CUDA=True、MPS=True，torch auto | cuda 优先 |
| DEV-003 | P0 | 新增 | CUDA=False、MPS=True | mps |
| DEV-004 | P0 | 新增 | CUDA=False、MPS=False | cpu |
| DEV-005 | P0 | 新增 | 显式 CUDA 可用/不可用 | 返回/错误 |
| DEV-006 | P0 | 新增 | 显式 MPS 可用/不可用 | 返回/错误 |
| DEV-007 | P0 | 部分现有 | 当前未固定硬件，只断言 ctranslate2 auto 结果为 cpu/cuda 且非 mps | 永不选择 mps |
| DEV-008 | P0 | 现有 | ctranslate2 mps | 错误和建议 |
| DEV-009 | P0 | 新增 | ctranslate2 CUDA | cuda |
| DEV-010 | P0 | 新增 | 未知 preference | 错误列支持项 |
| DEV-011 | P0 | 新增 | preference=auto + 未知 backend | DeviceNotAvailableError |
| DEV-012 | P1 | 新增 | Pipeline 前后 config | 不原地污染或记录 resolved config |
| DEV-013 | P1 | 新增 | preference=cpu/cuda/mps + 未知 backend | 固定当前“跳过 backend 校验”行为，或在重构中统一改为拒绝并提供迁移说明 |
| DEV-014 | P0 | 新增 | Mock CUDA=False、MPS=True，ctranslate2 auto | 返回 cpu，证明只存在 MPS 时会回退 CPU |
| UTL-015 | P1 | 现有 | 三种时长 | 格式正确 |
| UTL-016 | P2 | 新增 | 1h+/负数/NaN | 定义明确 |
| UTL-017 | P1 | 新增 | 从工作目录传入合法相对 WAV 路径 | 返回正确绝对路径 |

### 17.3 输出与文件系统

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| OUT-001 | P0 | 现有 | song.mp3 + root | 默认路径正确 |
| OUT-002 | P0 | 现有 | lyrics_format=txt | lyrics.txt |
| OUT-003 | P0 | 现有 | base 存在、无覆盖 | OutputExistsError |
| OUT-004 | P0 | 部分现有 | 当前只验证 `resolve_output_paths(..., overwrite=True)` 不拒绝；没有事务实现 | 当前路径解析允许覆盖；事务替换能力由 OUT-006/008/009 新增验证 |
| OUT-005 | P0 | 现有 | ensure dirs | 必要目录创建 |
| OUT-006 | P0 | 重写 | 旧目录+覆盖成功/失败 | 成功替换；失败回滚 |
| OUT-007 | P0 | 新增 | base 是文件 | 错误且文件不受损 |
| OUT-008 | P0 | 新增 | staging 中途失败 | 正式目录不变 |
| OUT-009 | P0 | 新增 | rename 失败 | 可回滚 |
| OUT-010 | P1 | 新增 | 同 stem 不同扩展输入 | 冲突提示固定 |
| OUT-011 | P1 | 新增 | 多级 root 不存在 | 创建正确 |
| OUT-012 | P1 | 新增 | root 无写权限 | 明确异常、无半目录 |
| OUT-013 | P1 | 新增 | 各 stem/格式名 | 命名正确 |
| OUT-014 | P1 | 现有 | 两类 viz 名 | 命名正确 |
| OUT-015 | P1 | 新增 | stem 名含 `../` | 拒绝路径穿越 |
| OUT-016 | P1 | 新增 | 并发同 song | 单一发布/明确冲突 |
| OUT-017 | P2 | 新增 | 可选阶段禁用 | 目录契约固定 |
| OUT-018 | P2 | 新增 | 非 ASCII song | 可创建和读取 |

### 17.4 Demucs 分离

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| SEP-001 | P0 | 重写 | 初始化 engine | `_model is None` |
| SEP-002 | P0 | 新增 | get_model；两次 load | 只加载一次 |
| SEP-003 | P0 | 新增 | demucs import 失败 | ModelNotFoundError |
| SEP-004 | P0 | 新增 | get_model 失败 | ModelNotFoundError |
| SEP-005 | P0 | 新增 | model.to 失败 | 统一项目异常 |
| SEP-006 | P0 | 新增 | 模型元数据 | get_model_info 非 None且正确 |
| SEP-007 | P0 | 重写 | Mock load_track/apply_model | 参数完整正确 |
| SEP-008 | P0 | 新增 | 检查 shape/device | `[1,C,T]` 和正确设备 |
| SEP-009 | P0 | 新增 | DC offset + 可控输出 | 官方归一化一致 |
| SEP-010 | P0 | 新增 | 静音 | 无 NaN/Inf |
| SEP-011 | P0 | 新增 | 极低方差 | 无数值爆炸 |
| SEP-012 | P0 | 新增 | 4 sources | 键/tensor/路径对应 |
| SEP-013 | P0 | 新增 | 6 轨模型 | 全部 sources 输出 |
| SEP-014 | P0 | 新增 | two_stems=vocals | vocals/no_vocals |
| SEP-015 | P0 | 新增 | 目标+残差 | 数值契约正确 |
| SEP-016 | P0 | 新增 | two_stems 不存在 | 推理前错误 |
| SEP-017 | P0 | 重写 | stems 两轨 | 只保存两轨 |
| SEP-018 | P0 | 新增 | stems 含未知 | 不静默 |
| SEP-019 | P0 | 新增 | 全部不匹配 | 不得 0 轨静默成功 |
| SEP-020 | P0 | 新增 | CUDA OOM | AudioProcessingError+建议 |
| SEP-021 | P0 | 新增 | load_track 错误 | AudioProcessingError |
| SEP-022 | P0 | 新增 | apply_model 错误 | AudioProcessingError |
| SEP-023 | P0 | 新增 | apply/load import 失败 | AudioProcessingError |
| SEP-024 | P0 | 新增 | 某 stem 保存失败 | 指明 stem；不发布半成品 |
| SEP-025 | P0 | 新增 | 1D tensor | 变 `[1,T]` |
| SEP-026 | P0 | 新增 | `[1,C,T]` | squeeze 后仍二维 |
| SEP-027 | P0 | 新增 | WAV Mock writer | PCM_S/16-bit |
| SEP-028 | P0 | 新增 | FLAC Mock writer | format=flac |
| SEP-029 | P0 | 新增 | MP3 合法 bitrate | 整数 bitrate |
| SEP-030 | P0 | 新增 | MP3 writer import 失败 | 回退 torchaudio |
| SEP-031 | P1 | 新增 | MP3 codec 运行时失败 | 回退或明确 codec 错误 |
| SEP-032 | P1 | 新增 | 无 torchaudio但 Demucs MP3 可用 | 不被无关 import 阻断 |
| SEP-033 | P1 | 新增 | 无效 bitrate | 推理前拒绝 |
| SEP-034 | P1 | 新增 | 直接 API 非法 format | 不静默按 WAV |
| SEP-035 | P1 | 新增 | quiet/non-TTY progress | 符合 CLI 契约 |
| SEP-036 | P1 | 新增 | 同 engine 多次 separate | 模型复用、结果隔离 |
| SEP-037 | P1 | 新增 | CPU/GPU tensor | 写前安全搬 CPU |
| SEP-038 | P1 | 新增 | 非预期 shape/source 数 | 结构化错误 |
| SEP-039 | P1 | 新增 | NaN/Inf/超幅 | 检测/裁剪/拒绝策略固定 |
| SEP-040 | P2 slow | 新增 | 官方 Demucs CLI 对比 | 每轨误差在阈值内 |
| SEP-041 | P2 slow | 新增 | 真实 htdemucs CPU | 文件可读、元数据正确 |
| SEP-042 | P2 slow | 新增 | 真实 htdemucs_6s | 6 轨完整 |
| SEP-043 | P2 slow | 新增 | 真实三种输出 codec | 可读、时长正确 |
| SEP-044 | P2 slow | 新增 | 重混全部 stems | 误差指标有基线 |

### 17.5 歌词

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| LYR-001 | P0 | 现有 | 初始化 | 惰性 model |
| LYR-002 | P0 | 新增 | device=cpu | int8 |
| LYR-003 | P0 | 新增 | device=cuda | float16 |
| LYR-004 | P0 | 新增 | mps | 明确错误 |
| LYR-005 | P0 | 新增 | import 失败 | TranscriptionError |
| LYR-006 | P0 | 新增 | 模型构造失败 | TranscriptionError |
| LYR-007 | P0 | 新增 | 多次 get_model | 只创建一次 |
| LYR-008 | P0 | 新增 | Mock iterator/info | 固定参数和 language 透传 |
| LYR-009 | P0 | 新增 | 文本两侧空格 | strip |
| LYR-010 | P0 | 新增 | iterator 中途失败 | TranscriptionError |
| LYR-011 | P0 | 现有 | SRT | 序号/时间/文本 |
| LYR-012 | P0 | 新增 | LRC | `[mm:ss.xx]` |
| LYR-013 | P0 | 现有 | VTT | header/时间 |
| LYR-014 | P0 | 现有 | TXT | 无时间 |
| LYR-015 | P0 | 现有 | JSON | 字段/UTF-8 |
| LYR-016 | P0 | 现有 | 非法格式 | 错误 |
| LYR-017 | P0 | 现有 | 中文 | 不损坏 |
| LYR-018 | P1 | 新增 | 负时间 | clamp 0 |
| LYR-019 | P1 | 新增 | 1h+/边界 | 舍入契约固定 |
| LYR-020 | P1 | 新增 | end<start/NaN/Inf | 拒绝或规范化 |
| LYR-021 | P1 | 新增 | 空 segments | 五种空输出定义明确 |
| LYR-022 | P1 | 新增 | 缺字段 | 统一 TranscriptionError |
| LYR-023 | P1 | 新增 | 父目录不存在 | 创建或明确错误 |
| LYR-024 | P1 | 新增 | OSError | TranscriptionError |
| LYR-025 | P2 | 新增 | JSON schema 版本 | 兼容或含语言元数据 |
| LYR-026 | P2 slow | 新增 | Whisper tiny 真样本 | 非空结果、语言有效 |
| LYR-027 | P2 slow | 新增 | 指定/自动语言 | 行为可复现 |

### 17.6 可视化

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| VIZ-001 | P0 | 现有 | 三类型+vocals | 3 PNG |
| VIZ-002 | P0 | 现有 | stem=None | `_original` |
| VIZ-003 | P0 | 现有 | 仅 waveform | 1 张 |
| VIZ-004 | P0 | 部分现有 | 当前只断言抛出任意 `Exception`；新增精确类型断言 | 不存在音频抛 `VisualizationError` |
| VIZ-005 | P0 | 新增 | Mock librosa.load | sr=None，每次 generate 一次 |
| VIZ-006 | P0 | 新增 | waveform 失败 | 后两种继续 |
| VIZ-007 | P0 | 新增 | 未知 type | warning 跳过 |
| VIZ-008 | P0 | 新增 | savefig 失败 | figure 关闭 |
| VIZ-009 | P1 | 新增 | 正常全量 | 无未关闭 figure |
| VIZ-010 | P1 | 新增 | 读取 PNG | 非空、尺寸正确 |
| VIZ-011 | P1 | 新增 | mono/stereo | 均可生成 |
| VIZ-012 | P1 | 新增 | output_dir 不存在 | 契约明确 |
| VIZ-013 | P1 | 新增 | 空/极短信号 | 明确结果 |
| VIZ-014 | P1 | 新增 | NaN/Inf | 明确策略 |
| VIZ-015 | P2 | 新增 | N stems × M types | 数量/命名正确 |
| VIZ-016 | P2 slow | 新增 | 30 分钟音频 | 内存/耗时基线 |
| VIZ-017 | P2 slow | 新增 | 批量 stems | 无资源泄漏 |

### 17.7 Pipeline

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| PIP-001 | P0 | 部分现有 | 当前只断言 input_file 和空 stems；新增 Mock 断言 | 校验后不创建目录、不构造分离引擎、不执行分离 |
| PIP-002 | P0 | 现有 | 不存在输入 | FileNotFoundError |
| PIP-003 | P0 | 现有 | Mock 4 stems | 结果正确 |
| PIP-004 | P0 | 新增 | Mock 顺序 | validate→device→paths→dirs→separate |
| PIP-005 | P0 | 新增 | 输出冲突 | 不加载模型 |
| PIP-006 | P0 | 新增 | 分离失败 | 向上传播 |
| PIP-007 | P0 | 新增 | 有 vocals+lyrics | 使用 vocals |
| PIP-008 | P0 | 新增 | 无 vocals+lyrics | 使用原音频 |
| PIP-009 | P0 | 新增 | lyrics disabled | 不构造 Whisper |
| PIP-010 | P0 | 新增 | lyrics 成功 | lyrics_path 正确 |
| PIP-011 | P0 | 新增 | Whisper device 失败 | 记录并继续 |
| PIP-012 | P0 | 新增 | transcribe/save 失败 | stems 保留、stage failed |
| PIP-013 | P0 | 新增 | viz disabled | 不构造 Visualizer |
| PIP-014 | P0 | 新增 | 2 stems×3 types | 原音频先、共 9 张 |
| PIP-015 | P0 | 新增 | 原音频 viz load 失败 | 分离仍成功 |
| PIP-016 | P0 | 新增 | 单图失败 | 其他图保留 |
| PIP-017 | P0 | 新增 | lyrics 失败/viz 成功 | 状态独立 |
| PIP-018 | P0 | 新增 | lyrics 成功/viz 失败 | 歌词保留 |
| PIP-019 | P0 | 新增 | 两可选阶段都失败 | 部分成功契约固定 |
| PIP-020 | P0 | 新增 | overwrite+分离失败 | 旧输出不丢 |
| PIP-021 | P1 | 新增 | auto config | 不原地污染 |
| PIP-022 | P1 | 新增 | 同 Pipeline 两次 run | 状态不串扰 |
| PIP-023 | P1 | 新增 | 分离返回空 | 不静默成功 |
| PIP-024 | P1 | 新增 | summary | 与结果一致 |
| PIP-025 | P1 | 新增 | quiet/verbose | 输出协议正确 |
| PIP-026 | P1 | 新增 | KeyboardInterrupt | 清 staging |
| PIP-027 | P1 | 新增 | SIGTERM 恢复 | 下次可清理 |
| PIP-028 | P2 | 新增 | StageResult | status/error/time 完整 |
| PIP-029 | P2 | 新增 | strict 可选阶段失败 | 整体非 0 |
| PIP-030 | P2 slow | 新增 | 短 WAV 全流程 | 所有产物可读 |

### 17.8 CLI

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| CLI-001 | P0 | 现有 | `--help` | 子命令完整 |
| CLI-002 | P0 | 现有 | `--version` | 0.1.0 |
| CLI-003 | P0 | 现有 | separate help | 参数完整 |
| CLI-004 | P0 | 新增 | 每个参数+Mock merge | 映射正确 |
| CLI-005 | P0 | 新增 | 重复 stems | list 顺序正确 |
| CLI-006 | P0 | 新增 | 重复 viz-types | list 顺序正确 |
| CLI-007 | P0 | 新增 | 三组正反 bool | True/False/None 正确 |
| CLI-008 | P0 | 现有 | dry-run | 不构造 Pipeline |
| CLI-009 | P0 | 现有 | dry-run+可选阶段 | 计划完整 |
| CLI-010 | P0 | 新增 | dry-run+坏扩展/设备/冲突 | 最终语义固定 |
| CLI-011 | P0 | 新增 | 非 dry-run Pipeline 成功 | exit 0/参数正确 |
| CLI-012 | P0 | 新增 | OutputExistsError | exit 1 |
| CLI-013 | P0 | 新增 | 处理异常 | 中文错误+exit 1 |
| CLI-014 | P0 | 新增 | 配置异常 | 中文配置错误+exit 1 |
| CLI-015 | P0 | 现有 | 非法模型 | 非 0 |
| CLI-016 | P0 | 现有 | 输入不存在 | 非 0 |
| CLI-017 | P0 | 新增 | 默认命令别名 | 与 separate 等价 |
| CLI-018 | P0 | 新增 | python -m | 与 console script 等价 |
| CLI-019 | P1 | 扩展 | models | 6 模型和描述正确 |
| CLI-020 | P1 | 新增 | 真实缓存布局 Mock | installed 准确 |
| CLI-021 | P1 | 新增 | installed-only 有/无 | 列表/计数正确 |
| CLI-022 | P1 | 现有 | info 不存在 | 非 0 |
| CLI-023 | P1 | 新增 | mono/stereo WAV | 元信息正确 |
| CLI-024 | P1 | 新增 | 多声道 | N 声道文案 |
| CLI-025 | P1 | 新增 | 损坏音频/codec 缺失 | 中文统一错误 |
| CLI-026 | P1 | 新增 | quiet | INFO 抑制 |
| CLI-027 | P1 | 新增 | verbose | DEBUG且不重复 |
| CLI-028 | P1 | 新增 | Unicode 路径 | 成功 |
| CLI-029 | P2 | 新增 | README 示例 smoke | 语法全部有效 |

### 17.9 集成、E2E、兼容性与性能

| ID | 优先级 | 状态 | 输入/步骤或 Mock | 预期结果 |
|---|---|---|---|---|
| INT-001 | P0 | 新增 | 最小 CPU 安装 import | 无缺失直接依赖 |
| INT-002 | P0 | 新增 | wheel 干净安装 | 两种入口可用 |
| INT-003 | P0 | 新增 | 无 torchaudio | 安装/运行前明确诊断 |
| INT-004 | P0 | 新增 | 无 ffmpeg/torchcodec | codec 诊断可执行 |
| INT-005 | P1 | 新增 | Python 3.10/3.11/3.12 | 测试和 smoke 通过 |
| INT-006 | P2 | 新增 | Python 3.13 | 明确支持/排除 |
| INT-007 | P1 | 新增 | Demucs 4.0.1 | 低层 API 通过 |
| INT-008 | P1 | 新增 | 允许区间最新版 Demucs | 兼容或锁上界 |
| INT-009 | P1 | 新增 | Linux CPU | 核心通过 |
| INT-010 | P1 | 新增 | Linux CUDA | Demucs/Whisper CUDA |
| INT-011 | P1 | 新增 | macOS MPS | Demucs MPS、Whisper CPU |
| INT-012 | P2 | 新增 | Windows CPU/CUDA | 路径/codec 通过 |
| INT-013 | P1 | 新增 | 7 种真实输入 | 解码或条件说明 |
| INT-014 | P1 | 新增 | 3 种真实输出 | 元数据和时长正确 |
| E2E-001 | P1 slow | 新增 | 默认 htdemucs | 4 轨完整 |
| E2E-002 | P1 slow | 新增 | two-stems vocals | 两轨正确 |
| E2E-003 | P1 slow | 新增 | stems 两轨 | 只输出两轨 |
| E2E-004 | P1 slow | 新增 | lyrics LRC+tiny | LRC 可解析 |
| E2E-005 | P1 slow | 新增 | waveform only | 原音频+每 stem |
| E2E-006 | P1 slow | 新增 | 全功能 | 全部产物/摘要一致 |
| E2E-007 | P1 slow | 新增 | 重复无 overwrite | 第二次拒绝、首次不变 |
| E2E-008 | P1 slow | 新增 | overwrite 第二次失败 | 首次可恢复 |
| E2E-009 | P2 slow | 新增 | Whisper 冷/热缓存 | 复用正确 |
| E2E-010 | P2 slow | 新增 | Demucs 冷/热缓存 | 复用和 models 准确 |
| PERF-001 | P2 slow | 新增 | 1/5/30 分钟 CPU | 耗时/RSS/大小基线 |
| PERF-002 | P2 slow | 新增 | CUDA/MPS/CPU | 速度/显存基线 |
| PERF-003 | P2 slow | 新增 | shifts/overlap 矩阵 | 质量/耗时曲线 |
| PERF-004 | P2 slow | 新增 | 原音频+6 stems 图 | 内存/figure 稳定 |
| PERF-005 | P2 slow | 新增 | 同进程 20 首 | 无内存/句柄泄漏 |
| QUAL-001 | P2 slow | 新增 | golden vs 官方 Demucs | 波形/SDR 阈值 |
| QUAL-002 | P2 slow | 新增 | 重混 stems | 响度/波形偏差受控 |
| QUAL-003 | P2 slow | 新增 | vocals Whisper | WER/CER 不退化 |
| SEC-001 | P1 | 新增 | 恶意名/符号链接 | 无路径穿越 |
| SEC-002 | P1 | 新增 | overwrite 符号链接/文件 | 无越界删除 |
| REL-001 | P1 | 新增 | PR CI | pytest/ruff/mypy/build 全过 |
| REL-002 | P1 | 新增 | 覆盖率 | 核心 branch >=90%，总体 >=85% |
| REL-003 | P2 | 新增 | 发布 smoke | wheel help/version/dry-run/info |

---

## 18. 测试分层与 Mock 边界

### 单元层

不下载模型、不访问 GPU、不依赖 ffmpeg：配置、路径、设备 Mock、Demucs 导入边界 Mock、WhisperModel Mock、歌词 writer、小数组可视化、依赖注入 Pipeline、Mock Pipeline 的 Typer CLI。

### 集成层

真实 torchaudio/soundfile/librosa/matplotlib，使用 0.1–1 秒合成音频；wheel 干净安装；Demucs/Whisper 放到 slow marker；codec 能力使用显式 marker。

### 端到端层

使用许可明确的短音频，固定模型版本/checksum，以 CPU 为最低可复现环境；GPU/MPS 独立矩阵。验证文件可读、目录结构、时长、采样率、声道和数值范围，不只检查存在。

### 推荐 fixture

- stereo/mono/silent/offset/low-variance WAV；
- 授权短人声；
- fake Demucs model 和 `[1,S,C,T]` tensor；
- fake Whisper iterator/info；
- output transaction 临时目录；
- codec/device capability fixtures。

---

## 19. 推荐重构目标架构

```text
music_sep/
├── domain/
│   ├── config.py              # Raw/Validated/ResolvedConfig
│   ├── results.py             # PipelineResult、StageResult、Artifact
│   └── errors.py              # 稳定异常码
├── application/
│   ├── pipeline.py            # 纯编排
│   └── planning.py            # dry-run/validate-only
├── ports/
│   ├── separator.py
│   ├── transcriber.py
│   ├── visualizer.py
│   ├── audio_writer.py
│   └── output_store.py
├── adapters/
│   ├── demucs_separator.py
│   ├── whisper_transcriber.py
│   ├── librosa_visualizer.py
│   ├── torchaudio_writer.py
│   └── filesystem_output.py
├── cli.py
└── compatibility.py
```

建议引入 Protocol 接口、事务输出和结构化 `StageResult(status=success|skipped|failed, artifacts, error_code, message, elapsed)`。CLI 默认可继续兼容“可选阶段失败但整体成功”，但必须在摘要中明确“部分成功”。

配置应区分 RawConfig、ValidatedConfig 和 ResolvedConfig，避免 Pipeline 原地修改输入配置。

---

## 20. 推荐重构实施顺序

### Phase 0：冻结基线

锁依赖；重写 separation 测试；建立归一化 P0 测试；保存小型合法音频/golden；建立 pytest+ruff+mypy+wheel CI。

**Gate 0：**测试基线可信，P0 characterization tests 稳定。

### Phase 1：配置与领域模型

严格 schema；固定优先级/bool/list；Raw/Validated/Resolved；兼容旧 dataclass；决定未知字段、空 stems 和设备策略。

**Gate 1：**配置和 CLI 映射全绿。

### Phase 2：Demucs

对照 4.0.1 修归一化；抽 adapter；推理前校验 sources；抽 audio writer；处理 shape/bitrate/codec/OOM；与官方 CLI 数值对比。

**Gate 2：**SEP P0/P1 和真实 golden 全绿。

### Phase 3：事务输出

staging、临时文件、原子 commit、overwrite 回滚、中断恢复、路径安全。

**Gate 3：**任意注入失败不破坏旧正式产物。

### Phase 4：歌词与可视化

抽 Whisper/Visualizer；固定 JSON 策略；segment schema；优化长音频；保留局部失败。

**Gate 4：**格式、时间戳、图片和资源测试全绿。

### Phase 5：Pipeline 与 CLI

依赖注入；StageResult；统一 dry-run；默认别名决策；统一 exit/quiet/progress/错误；修 models/info。

**Gate 5：**CLI、Pipeline、文档示例全绿。

### Phase 6：兼容与发布

旧 import 兼容；旧/新输出比较；README smoke；迁移说明；干净 wheel；多平台矩阵。

**Gate 6：**无未解释行为差异。

---

## 21. 重构完成验收标准

### 功能

- [ ] 7 种输入扩展有明确支持条件；
- [ ] 6 个模型通过配置和 smoke；
- [ ] 默认 4 轨、6 轨、two-stems、stems 正确；
- [ ] WAV/FLAC/MP3 可独立解码；
- [ ] 五种歌词格式可验证；
- [ ] 原音频和每 stem 三种图正确；
- [ ] models/info/version 明确；
- [ ] 配置优先级、输出命名兼容；
- [ ] 可选阶段失败有结构化部分成功结果。

### 正确性

- [ ] RISK-AUD-001 已用官方实现证明修复；
- [ ] 重混误差满足阈值；
- [ ] two-stems 数值关系正确；
- [ ] 无 NaN/Inf/DC 污染；
- [ ] 无 0 stem 静默成功；
- [ ] 时间戳和图片资源稳定。

### 可靠性

- [ ] overwrite 失败不丢旧产物；
- [ ] 中途失败不发布残缺目录；
- [ ] Ctrl+C/SIGTERM 可恢复；
- [ ] 并发同名有锁/冲突；
- [ ] Unicode/长路径通过；
- [ ] 无路径穿越或越界删除。

### 工程质量

- [ ] pytest、Ruff、mypy 全绿；
- [ ] 核心 branch coverage >=90%，总体 >=85%；
- [ ] wheel 构建和干净安装通过；
- [ ] Python/OS/设备矩阵通过；
- [ ] 依赖锁定、CI、发布 smoke 完成；
- [ ] 中英文 README 与实际 CLI 一致。

---

## 22. 开发者快速定位索引

| 能力 | 当前文件 |
|---|---|
| 命令/参数/退出码 | `src/music_sep/cli.py` |
| 默认值/TOML/校验 | `src/music_sep/config.py` |
| 模型/格式常量 | `src/music_sep/constants.py` |
| Demucs | `src/music_sep/separation.py` |
| Whisper/歌词 | `src/music_sep/lyrics.py` |
| 图像 | `src/music_sep/visualization.py` |
| 编排/降级 | `src/music_sep/pipeline.py` |
| 输出/覆盖 | `src/music_sep/outputs.py` |
| 输入/设备/日志 | `src/music_sep/utils.py` |
| 异常 | `src/music_sep/exceptions.py` |
| 安装/入口 | `pyproject.toml` |
| 文档 | `README.md`、`README_CN.md` |
| 测试 | `tests/` |

---

## 23. 最终交接结论

重构风险集中在四点：

1. 当前 Demucs 归一化逻辑存在 P0 数值缺陷；
2. overwrite 会先删旧目录，输出没有事务保护；
3. 核心分离模块的现有 3 个测试全部针对旧 API；
4. README 默认命令、功能独立性和依赖声明与真实实现不一致。

最安全的顺序是：

```text
可信 characterization tests
→ 修复音频正确性
→ 抽取第三方 adapter
→ 引入事务输出
→ 重写 Pipeline
→ 最后替换 CLI 外壳
```

持续执行第 17 节测试矩阵和第 21 节验收标准，即可区分必须保留的产品能力、偶然实现细节和不能复制的缺陷。
