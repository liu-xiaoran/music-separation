from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar

from music_sep.domain.config import RawConfig, ValidatedConfig, validate_raw_config
from music_sep.domain.errors import ConfigValidationError
from music_sep.exceptions import ConfigurationError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger("music_sep")
T = TypeVar("T")


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


_KNOWN_FIELDS = {
    "separation": frozenset({"model", "device", "shifts", "overlap", "two_stems", "stems"}),
    "lyrics": frozenset(
        {"enabled", "whisper_model", "whisper_device", "language", "output_format"}
    ),
    "visualization": frozenset({"enabled", "types"}),
    "output": frozenset({"output_dir", "format", "bitrate", "overwrite"}),
}


def load_toml_config(path: Path) -> dict[str, Any]:
    """加载 TOML 配置文件。"""

    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError:
        raise ConfigurationError(f"配置文件不存在: {path}") from None
    except Exception as exc:
        raise ConfigurationError(f"配置文件解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("配置文件顶层必须是 TOML 表")
    return data


def _merge_field(default_value: T, toml_value: object | None, cli_value: object | None) -> object:
    """三层合并：CLI 非 None > TOML 非 None > 默认值。"""

    if cli_value is not None:
        return cli_value
    if toml_value is not None:
        return toml_value
    return default_value


def _warn_unknown_toml(data: dict[str, Any]) -> None:
    for section_name in data:
        if section_name not in _KNOWN_FIELDS:
            logger.warning("忽略未知配置节: [%r]", section_name)

    for section_name, known_fields in _KNOWN_FIELDS.items():
        section = data.get(section_name)
        if not isinstance(section, dict):
            continue
        for field_name in section:
            if field_name not in known_fields:
                logger.warning(
                    "忽略未知配置字段: [%r].%r",
                    section_name,
                    field_name,
                )


def _get_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise ConfigurationError(f"配置节 [{name}] 类型错误: 期望表，得到 {type(section).__name__}")
    return section


def _raw_from_app(config: AppConfig) -> RawConfig:
    if not isinstance(config, AppConfig):
        raise ConfigValidationError(
            f"config 类型错误: 期望 AppConfig，得到 {type(config).__name__}"
        )
    if not isinstance(config.separation, SeparationConfig):
        raise ConfigValidationError("config.separation 必须是 SeparationConfig")
    if not isinstance(config.lyrics, LyricsConfig):
        raise ConfigValidationError("config.lyrics 必须是 LyricsConfig")
    if not isinstance(config.visualization, VisualizationConfig):
        raise ConfigValidationError("config.visualization 必须是 VisualizationConfig")
    if not isinstance(config.output, OutputConfig):
        raise ConfigValidationError("config.output 必须是 OutputConfig")

    return RawConfig(
        separation_model=config.separation.model,
        separation_device=config.separation.device,
        separation_shifts=config.separation.shifts,
        separation_overlap=config.separation.overlap,
        separation_two_stems=config.separation.two_stems,
        separation_stems=config.separation.stems,
        lyrics_enabled=config.lyrics.enabled,
        whisper_model=config.lyrics.whisper_model,
        whisper_device=config.lyrics.whisper_device,
        lyrics_language=config.lyrics.language,
        lyrics_format=config.lyrics.output_format,
        visualization_enabled=config.visualization.enabled,
        visualization_types=config.visualization.types,
        output_dir=config.output.output_dir,
        output_format=config.output.format,
        output_bitrate=config.output.bitrate,
        output_overwrite=config.output.overwrite,
    )


def _raw_from_validated(config: ValidatedConfig) -> RawConfig:
    return RawConfig(
        separation_model=config.separation_model,
        separation_device=config.separation_device,
        separation_shifts=config.separation_shifts,
        separation_overlap=config.separation_overlap,
        separation_two_stems=config.separation_two_stems,
        separation_stems=config.separation_stems,
        lyrics_enabled=config.lyrics_enabled,
        whisper_model=config.whisper_model,
        whisper_device=config.whisper_device,
        lyrics_language=config.lyrics_language,
        lyrics_format=config.lyrics_format,
        visualization_enabled=config.visualization_enabled,
        visualization_types=config.visualization_types,
        output_dir=config.output_dir,
        output_format=config.output_format,
        output_bitrate=config.output_bitrate,
        output_overwrite=config.output_overwrite,
    )


def to_validated_config(config: AppConfig | ValidatedConfig) -> ValidatedConfig:
    """创建不信任调用方 collection 或类型标注的已校验领域快照。"""

    try:
        raw = (
            _raw_from_validated(config)
            if isinstance(config, ValidatedConfig)
            else _raw_from_app(config)
        )
        return validate_raw_config(raw)
    except ConfigValidationError as exc:
        raise ConfigurationError(str(exc)) from exc


def to_app_config(
    config: ValidatedConfig,
    *,
    separation_device: str | None = None,
    whisper_device: str | None = None,
) -> AppConfig:
    """从领域快照创建不共享 collection 的兼容 AppConfig。"""

    return AppConfig(
        separation=SeparationConfig(
            model=config.separation_model,
            device=(
                separation_device if separation_device is not None else config.separation_device
            ),
            shifts=config.separation_shifts,
            overlap=config.separation_overlap,
            two_stems=config.separation_two_stems,
            stems=(list(config.separation_stems) if config.separation_stems is not None else None),
        ),
        lyrics=LyricsConfig(
            enabled=config.lyrics_enabled,
            whisper_model=config.whisper_model,
            whisper_device=(
                whisper_device if whisper_device is not None else config.whisper_device
            ),
            language=config.lyrics_language,
            output_format=config.lyrics_format,
        ),
        visualization=VisualizationConfig(
            enabled=config.visualization_enabled,
            types=list(config.visualization_types),
        ),
        output=OutputConfig(
            output_dir=config.output_dir,
            format=config.output_format,
            bitrate=config.output_bitrate,
            overwrite=config.output_overwrite,
        ),
    )


def merge_config(
    toml_path: Optional[Path] = None,
    model: Optional[str] = None,
    device: Optional[str] = None,
    shifts: Optional[int] = None,
    overlap: Optional[float] = None,
    two_stems: Optional[str] = None,
    stems: Optional[list[str]] = None,
    lyrics_enabled: Optional[bool] = None,
    whisper_model: Optional[str] = None,
    whisper_device: Optional[str] = None,
    language: Optional[str] = None,
    lyrics_format: Optional[str] = None,
    visualize_enabled: Optional[bool] = None,
    viz_types: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    output_format: Optional[str] = None,
    bitrate: Optional[str] = None,
    overwrite: Optional[bool] = None,
) -> AppConfig:
    """按 CLI > TOML > 默认值合并并校验配置。"""

    toml_data = load_toml_config(toml_path) if toml_path is not None else {}
    _warn_unknown_toml(toml_data)
    toml_sep = _get_section(toml_data, "separation")
    toml_lyrics = _get_section(toml_data, "lyrics")
    toml_visualization = _get_section(toml_data, "visualization")
    toml_output = _get_section(toml_data, "output")

    raw = RawConfig(
        separation_model=_merge_field("htdemucs", toml_sep.get("model"), model),
        separation_device=_merge_field("auto", toml_sep.get("device"), device),
        separation_shifts=_merge_field(1, toml_sep.get("shifts"), shifts),
        separation_overlap=_merge_field(0.25, toml_sep.get("overlap"), overlap),
        separation_two_stems=_merge_field(
            None,
            toml_sep.get("two_stems"),
            two_stems,
        ),
        separation_stems=_merge_field(None, toml_sep.get("stems"), stems),
        lyrics_enabled=_merge_field(
            False,
            toml_lyrics.get("enabled"),
            lyrics_enabled,
        ),
        whisper_model=_merge_field(
            "medium",
            toml_lyrics.get("whisper_model"),
            whisper_model,
        ),
        whisper_device=_merge_field(
            "auto",
            toml_lyrics.get("whisper_device"),
            whisper_device,
        ),
        lyrics_language=_merge_field(
            None,
            toml_lyrics.get("language"),
            language,
        ),
        lyrics_format=_merge_field(
            "srt",
            toml_lyrics.get("output_format"),
            lyrics_format,
        ),
        visualization_enabled=_merge_field(
            False,
            toml_visualization.get("enabled"),
            visualize_enabled,
        ),
        visualization_types=_merge_field(
            ("waveform", "spectrogram", "mel"),
            toml_visualization.get("types"),
            viz_types,
        ),
        output_dir=_merge_field(
            Path("demo"),
            toml_output.get("output_dir"),
            output_dir,
        ),
        output_format=_merge_field(
            "wav",
            toml_output.get("format"),
            output_format,
        ),
        output_bitrate=_merge_field(
            "128k",
            toml_output.get("bitrate"),
            bitrate,
        ),
        output_overwrite=_merge_field(
            False,
            toml_output.get("overwrite"),
            overwrite,
        ),
    )

    try:
        return to_app_config(validate_raw_config(raw))
    except ConfigValidationError as exc:
        raise ConfigurationError(str(exc)) from exc


def validate_config(config: AppConfig) -> None:
    """校验公开兼容配置，不修改调用方对象。"""

    to_validated_config(config)
