from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from music_sep.domain.catalog import (
    DEMUCS_MODEL_DESCRIPTIONS,
    DEVICE_PREFERENCES,
    LYRICS_FORMATS,
    OUTPUT_FORMATS,
    RUNTIME_BACKENDS,
    VISUALIZATION_TYPES,
    WHISPER_MODELS,
)
from music_sep.domain.errors import ConfigValidationError, RuntimeResolutionError

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
ActualDevice = Literal["cpu", "cuda", "mps"]
RuntimeBackend = Literal["torch", "ctranslate2"]
_MP3_BITRATE_PATTERN = re.compile(r"^([1-9]\d{0,2})[kK]$")


def _freeze_collection(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


@dataclass(frozen=True, slots=True)
class RawConfig:
    """已完成优先级合并、尚未校验的配置快照。"""

    separation_model: object
    separation_device: object
    separation_shifts: object
    separation_overlap: object
    separation_two_stems: object
    separation_stems: object
    lyrics_enabled: object
    whisper_model: object
    whisper_device: object
    lyrics_language: object
    lyrics_format: object
    visualization_enabled: object
    visualization_types: object
    output_dir: object
    output_format: object
    output_bitrate: object
    output_overwrite: object

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "separation_stems",
            _freeze_collection(self.separation_stems),
        )
        object.__setattr__(
            self,
            "visualization_types",
            _freeze_collection(self.visualization_types),
        )


@dataclass(frozen=True, slots=True)
class ValidatedConfig:
    """类型、静态枚举和字段关系均已校验的逻辑配置。"""

    separation_model: str
    separation_device: DevicePreference
    separation_shifts: int
    separation_overlap: float
    separation_two_stems: str | None
    separation_stems: tuple[str, ...] | None
    lyrics_enabled: bool
    whisper_model: str
    whisper_device: DevicePreference
    lyrics_language: str | None
    lyrics_format: str
    visualization_enabled: bool
    visualization_types: tuple[str, ...]
    output_dir: Path
    output_format: str
    output_bitrate: str
    output_overwrite: bool


@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    """一次运行中某个第三方后端的请求设备与实际设备。"""

    requested: DevicePreference
    actual: ActualDevice
    backend: RuntimeBackend
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.requested not in DEVICE_PREFERENCES:
            raise RuntimeResolutionError(f"未知的设备类型: {self.requested!r}")
        if self.actual not in {"cpu", "cuda", "mps"}:
            raise RuntimeResolutionError(f"未知的实际设备类型: {self.actual!r}")
        if self.backend not in RUNTIME_BACKENDS:
            raise RuntimeResolutionError(f"未知的后端类型: {self.backend!r}")
        if self.backend == "ctranslate2" and self.actual == "mps":
            raise RuntimeResolutionError("ctranslate2 不支持 MPS")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()
        ):
            raise RuntimeResolutionError("设备回退原因必须是非空字符串或 None")

        if self.requested == "auto":
            if self.fallback_reason is not None and not (
                self.backend == "ctranslate2" and self.actual == "cpu"
            ):
                raise RuntimeResolutionError("auto 设备解析结果包含矛盾的回退原因")
            return

        if self.backend == "ctranslate2" and self.requested == "mps":
            if self.actual != "cpu" or self.fallback_reason is None:
                raise RuntimeResolutionError("ctranslate2 的显式 MPS 请求必须回退到 CPU 并说明原因")
            return

        if self.actual != self.requested:
            raise RuntimeResolutionError(f"显式请求 {self.requested!r} 不能解析为 {self.actual!r}")
        if self.fallback_reason is not None:
            raise RuntimeResolutionError("未发生设备回退时不应提供回退原因")

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """逻辑配置与本次运行设备解析结果。"""

    requested: ValidatedConfig
    separation: RuntimeResolution
    lyrics: RuntimeResolution | None

    def __post_init__(self) -> None:
        if self.separation.backend != "torch":
            raise RuntimeResolutionError("分离阶段必须使用 torch 后端")
        if self.separation.requested != self.requested.separation_device:
            raise RuntimeResolutionError("分离设备解析结果与请求配置不一致")
        if self.requested.lyrics_enabled:
            if self.lyrics is None:
                raise RuntimeResolutionError("已启用歌词阶段但缺少设备解析结果")
            if self.lyrics.backend != "ctranslate2":
                raise RuntimeResolutionError("歌词阶段必须使用 ctranslate2 后端")
            if self.lyrics.requested != self.requested.whisper_device:
                raise RuntimeResolutionError("歌词设备解析结果与请求配置不一致")
        elif self.lyrics is not None:
            raise RuntimeResolutionError("未启用歌词阶段时不应解析 Whisper 设备")


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError(
            f"{field_name} 类型错误: 期望字符串，得到 {type(value).__name__}"
        )
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ConfigValidationError(
            f"{field_name} 类型错误: 期望 bool，得到 {type(value).__name__}"
        )
    return value


def _validate_device(value: object, field_name: str) -> DevicePreference:
    device = _require_string(value, field_name)
    if device not in DEVICE_PREFERENCES:
        raise ConfigValidationError(
            f"{field_name} 未知的设备类型: {device!r}。支持: {' / '.join(DEVICE_PREFERENCES)}"
        )
    return device  # type: ignore[return-value]


def _validate_optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    result = _require_string(value, field_name)
    if not result:
        raise ConfigValidationError(f"{field_name} 不能为空字符串")
    return result


def _validate_string_tuple(
    value: object,
    field_name: str,
    *,
    allow_none: bool,
    allow_empty: bool,
) -> tuple[str, ...] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)):
        raise ConfigValidationError(f"{field_name} 必须是列表，当前类型: {type(value).__name__}")
    result = tuple(value)
    if not result and not allow_empty:
        raise ConfigValidationError(f"{field_name} 不能为空列表")
    for item in result:
        if not isinstance(item, str) or not item:
            raise ConfigValidationError(f"{field_name} 的每一项必须是非空字符串")
    return result


def _validate_mp3_bitrate(value: str) -> None:
    match = _MP3_BITRATE_PATTERN.fullmatch(value)
    if match is None:
        raise ConfigValidationError("output.bitrate MP3 比特率必须是形如 128k 的字符串")
    numeric = int(match.group(1))
    if not 8 <= numeric <= 320:
        raise ConfigValidationError("output.bitrate MP3 比特率必须在 8k 到 320k 之间")


def validate_raw_config(raw: RawConfig) -> ValidatedConfig:
    """校验 RawConfig 并返回不共享可变状态的领域快照。"""

    model = _require_string(raw.separation_model, "separation.model")
    if model not in DEMUCS_MODEL_DESCRIPTIONS:
        raise ConfigValidationError(
            f"未知的 Demucs 模型: {model!r}。可用模型: {', '.join(DEMUCS_MODEL_DESCRIPTIONS)}"
        )

    shifts = raw.separation_shifts
    if type(shifts) is not int or shifts < 1:
        raise ConfigValidationError(f"shifts 必须是 >= 1 的整数，当前值: {shifts!r}")

    overlap = raw.separation_overlap
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)):
        raise ConfigValidationError(f"overlap 必须是 0.0~1.0 的有限数，当前值: {overlap!r}")
    normalized_overlap = float(overlap)
    if not math.isfinite(normalized_overlap) or not 0.0 <= normalized_overlap <= 1.0:
        raise ConfigValidationError(f"overlap 必须是 0.0~1.0 的有限数，当前值: {overlap!r}")

    two_stems = _validate_optional_string(
        raw.separation_two_stems,
        "separation.two_stems",
    )
    stems = _validate_string_tuple(
        raw.separation_stems,
        "stems",
        allow_none=True,
        allow_empty=False,
    )
    if two_stems is not None and stems is not None:
        raise ConfigValidationError("--two-stems 和 --stems 不可同时使用")

    whisper_model = _require_string(raw.whisper_model, "lyrics.whisper_model")
    if whisper_model not in WHISPER_MODELS:
        raise ConfigValidationError(
            f"未知的 Whisper 模型: {whisper_model!r}。可用模型: {', '.join(WHISPER_MODELS)}"
        )

    lyrics_format = _require_string(raw.lyrics_format, "lyrics.output_format")
    if lyrics_format not in LYRICS_FORMATS:
        raise ConfigValidationError(
            f"不支持的歌词格式: {lyrics_format!r}。支持: {', '.join(LYRICS_FORMATS)}"
        )

    visualization_types = _validate_string_tuple(
        raw.visualization_types,
        "visualization.types",
        allow_none=False,
        allow_empty=True,
    )
    assert visualization_types is not None
    for visualization_type in visualization_types:
        if visualization_type not in VISUALIZATION_TYPES:
            raise ConfigValidationError(
                f"未知的可视化类型: {visualization_type!r}。支持: {', '.join(VISUALIZATION_TYPES)}"
            )

    output_format = _require_string(raw.output_format, "output.format")
    if output_format not in OUTPUT_FORMATS:
        raise ConfigValidationError(
            f"不支持的输出格式: {output_format!r}。支持: {', '.join(OUTPUT_FORMATS)}"
        )

    bitrate = _require_string(raw.output_bitrate, "output.bitrate")
    if output_format == "mp3":
        _validate_mp3_bitrate(bitrate)

    output_dir_value = raw.output_dir
    if not isinstance(output_dir_value, (str, Path)):
        raise ConfigValidationError(
            f"output.output_dir 类型错误: 期望字符串或 Path，得到 {type(output_dir_value).__name__}"
        )

    language = raw.lyrics_language
    if language is not None and not isinstance(language, str):
        raise ConfigValidationError(
            f"lyrics.language 类型错误: 期望字符串或 None，得到 {type(language).__name__}"
        )

    return ValidatedConfig(
        separation_model=model,
        separation_device=_validate_device(
            raw.separation_device,
            "separation.device",
        ),
        separation_shifts=shifts,
        separation_overlap=normalized_overlap,
        separation_two_stems=two_stems,
        separation_stems=stems,
        lyrics_enabled=_require_bool(raw.lyrics_enabled, "lyrics.enabled"),
        whisper_model=whisper_model,
        whisper_device=_validate_device(
            raw.whisper_device,
            "lyrics.whisper_device",
        ),
        lyrics_language=language,
        lyrics_format=lyrics_format,
        visualization_enabled=_require_bool(
            raw.visualization_enabled,
            "visualization.enabled",
        ),
        visualization_types=visualization_types,
        output_dir=Path(output_dir_value),
        output_format=output_format,
        output_bitrate=bitrate,
        output_overwrite=_require_bool(raw.output_overwrite, "output.overwrite"),
    )
