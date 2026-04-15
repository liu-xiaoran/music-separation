from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from music_sep.constants import AVAILABLE_MODELS, WHISPER_MODELS, VIZ_TYPES
from music_sep.exceptions import ConfigurationError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class SeparationConfig:
    model: str = "htdemucs"
    device: str = "auto"
    shifts: int = 1
    overlap: float = 0.25
    two_stems: Optional[str] = None
    stems: Optional[list[str]] = None


@dataclass
class LyricsConfig:
    enabled: bool = False
    whisper_model: str = "medium"
    whisper_device: str = "auto"
    language: Optional[str] = None
    output_format: str = "srt"


@dataclass
class VisualizationConfig:
    enabled: bool = False
    types: list[str] = field(default_factory=lambda: ["waveform", "spectrogram", "mel"])


@dataclass
class OutputConfig:
    output_dir: Path = Path("demo")
    format: str = "wav"
    bitrate: str = "128k"
    overwrite: bool = False


@dataclass
class AppConfig:
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    lyrics: LyricsConfig = field(default_factory=LyricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_toml_config(path: Path) -> dict:
    """加载 TOML 配置文件

    Args:
        path: 配置文件路径

    Returns:
        解析后的字典

    Raises:
        ConfigurationError: 文件不存在或格式错误
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"配置文件不存在: {path}")
    except Exception as e:
        raise ConfigurationError(f"配置文件解析失败: {e}")


def _merge_field(default_val, toml_val, cli_val):
    """三层合并：CLI > TOML > 默认值

    对于 Optional[bool]（三态），cli_val 为 None 表示未指定，回退到 toml_val。
    """
    if cli_val is not None:
        return cli_val
    if toml_val is not None:
        return toml_val
    return default_val


def merge_config(
    toml_path: Optional[Path] = None,
    # Separation
    model: Optional[str] = None,
    device: Optional[str] = None,
    shifts: Optional[int] = None,
    overlap: Optional[float] = None,
    two_stems: Optional[str] = None,
    stems: Optional[list[str]] = None,
    # Lyrics
    lyrics_enabled: Optional[bool] = None,
    whisper_model: Optional[str] = None,
    whisper_device: Optional[str] = None,
    language: Optional[str] = None,
    lyrics_format: Optional[str] = None,
    # Visualization
    visualize_enabled: Optional[bool] = None,
    viz_types: Optional[list[str]] = None,
    # Output
    output_dir: Optional[Path] = None,
    output_format: Optional[str] = None,
    bitrate: Optional[str] = None,
    overwrite: Optional[bool] = None,
) -> AppConfig:
    """三层配置合并：CLI > TOML > 默认值

    Args:
        toml_path: TOML 配置文件路径
        其余参数为 CLI 传入的值，None 表示未指定

    Returns:
        合并后的 AppConfig

    Raises:
        ConfigurationError: 配置值无效
    """
    # 加载 TOML 配置
    toml_data = {}
    if toml_path is not None:
        toml_data = load_toml_config(toml_path)

    toml_sep = toml_data.get("separation", {})
    toml_lyr = toml_data.get("lyrics", {})
    toml_viz = toml_data.get("visualization", {})
    toml_out = toml_data.get("output", {})

    # 校验 TOML section 类型必须为 dict
    for name, section in [
        ("separation", toml_sep), ("lyrics", toml_lyr),
        ("visualization", toml_viz), ("output", toml_out),
    ]:
        if not isinstance(section, dict):
            raise ConfigurationError(
                f"配置节 [{name}] 类型错误: 期望表，得到 {type(section).__name__}"
            )

    # 合并 Separation
    sep_config = SeparationConfig(
        model=_merge_field("htdemucs", toml_sep.get("model"), model),
        device=_merge_field("auto", toml_sep.get("device"), device),
        shifts=_merge_field(1, toml_sep.get("shifts"), shifts),
        overlap=_merge_field(0.25, toml_sep.get("overlap"), overlap),
        two_stems=_merge_field(None, toml_sep.get("two_stems"), two_stems),
        stems=_merge_field(None, toml_sep.get("stems"), stems),
    )

    # 合并 Lyrics
    lyr_config = LyricsConfig(
        enabled=_merge_field(False, toml_lyr.get("enabled"), lyrics_enabled),
        whisper_model=_merge_field("medium", toml_lyr.get("whisper_model"), whisper_model),
        whisper_device=_merge_field("auto", toml_lyr.get("whisper_device"), whisper_device),
        language=_merge_field(None, toml_lyr.get("language"), language),
        output_format=_merge_field("srt", toml_lyr.get("output_format"), lyrics_format),
    )

    # 合并 Visualization
    viz_config = VisualizationConfig(
        enabled=_merge_field(False, toml_viz.get("enabled"), visualize_enabled),
        types=_merge_field(
            ["waveform", "spectrogram", "mel"],
            toml_viz.get("types"),
            viz_types,
        ),
    )

    # 合并 Output — output_dir 必须是字符串或 Path
    toml_output_dir = toml_out.get("output_dir", "demo")
    if not isinstance(toml_output_dir, (str, Path)):
        raise ConfigurationError(
            f"output.output_dir 类型错误: 期望字符串，得到 {type(toml_output_dir).__name__}"
        )
    out_config = OutputConfig(
        output_dir=Path(_merge_field("demo", str(toml_output_dir), str(output_dir) if output_dir else None)),
        format=_merge_field("wav", toml_out.get("format"), output_format),
        bitrate=_merge_field("128k", toml_out.get("bitrate"), bitrate),
        overwrite=_merge_field(False, toml_out.get("overwrite"), overwrite),
    )

    config = AppConfig(
        separation=sep_config,
        lyrics=lyr_config,
        visualization=viz_config,
        output=out_config,
    )

    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """校验配置值的合法性

    Raises:
        ConfigurationError: 配置值无效
    """
    # 校验模型名
    if config.separation.model not in AVAILABLE_MODELS:
        raise ConfigurationError(
            f"未知的 Demucs 模型: {config.separation.model}。"
            f"可用模型: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    # 校验 Whisper 模型
    if config.lyrics.whisper_model not in WHISPER_MODELS:
        raise ConfigurationError(
            f"未知的 Whisper 模型: {config.lyrics.whisper_model}。"
            f"可用模型: {', '.join(WHISPER_MODELS)}"
        )

    # 校验歌词格式
    valid_lyrics_formats = {"srt", "vtt", "txt", "json"}
    if config.lyrics.output_format not in valid_lyrics_formats:
        raise ConfigurationError(
            f"不支持的歌词格式: {config.lyrics.output_format}。"
            f"支持: {', '.join(valid_lyrics_formats)}"
        )

    # 校验输出格式
    valid_output_formats = {"wav", "mp3", "flac"}
    if config.output.format not in valid_output_formats:
        raise ConfigurationError(
            f"不支持的输出格式: {config.output.format}。"
            f"支持: {', '.join(valid_output_formats)}"
        )

    # 校验可视化类型
    for vt in config.visualization.types:
        if vt not in VIZ_TYPES:
            raise ConfigurationError(
                f"未知的可视化类型: {vt}。支持: {', '.join(VIZ_TYPES)}"
            )

    # 校验互斥参数
    if config.separation.two_stems is not None and config.separation.stems is not None:
        raise ConfigurationError("--two-stems 和 --stems 不可同时使用")

    # 校验 stems 必须是 list（不能是字符串）
    if config.separation.stems is not None:
        if not isinstance(config.separation.stems, list):
            raise ConfigurationError(
                f"stems 必须是列表，当前类型: {type(config.separation.stems).__name__}"
            )

    # 校验 visualization.types 必须是 list
    if not isinstance(config.visualization.types, list):
        raise ConfigurationError(
            f"visualization.types 必须是列表，当前类型: {type(config.visualization.types).__name__}"
        )

    # 校验 overlap 范围
    if not 0.0 <= config.separation.overlap <= 1.0:
        raise ConfigurationError(f"overlap 必须在 0.0~1.0 之间，当前值: {config.separation.overlap}")

    # 校验 shifts
    if config.separation.shifts < 1:
        raise ConfigurationError(f"shifts 必须 >= 1，当前值: {config.separation.shifts}")
