import logging
import sys
from pathlib import Path
from typing import cast

from music_sep.adapters.torch_runtime import TorchRuntimeResolver
from music_sep.constants import SUPPORTED_INPUT_EXTENSIONS
from music_sep.domain.catalog import DEVICE_PREFERENCES, RUNTIME_BACKENDS
from music_sep.domain.config import (
    DevicePreference,
    RuntimeBackend,
    RuntimeResolution,
)
from music_sep.domain.errors import RuntimeResolutionError
from music_sep.exceptions import DeviceNotAvailableError, UnsupportedFormatError
from music_sep.ports.runtime import RuntimeResolver


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """配置日志

    Args:
        verbose: True=DEBUG级别
        quiet: True=ERROR级别
        两者都为True时，verbose优先

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger("music_sep")

    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.INFO

    logger.setLevel(level)

    # 避免重复添加 handler，但始终更新级别
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # 更新已有 handler 的级别
        for h in logger.handlers:
            h.setLevel(level)

    return logger


def validate_input_file(path: Path) -> Path:
    """验证输入文件存在且格式受支持

    Args:
        path: 输入文件路径

    Returns:
        绝对路径

    Raises:
        FileNotFoundError: 文件不存在
        UnsupportedFormatError: 不支持的格式
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"路径不是文件: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise UnsupportedFormatError(
            f"不支持的音频格式: {ext}。支持的格式: {', '.join(sorted(SUPPORTED_INPUT_EXTENSIONS))}"
        )

    return file_path


def resolve_device(
    preference: str = "auto",
    backend: str = "torch",
    *,
    resolver: RuntimeResolver | None = None,
) -> RuntimeResolution:
    """解析请求设备，并保留实际设备和可见回退原因。"""

    active_resolver = resolver if resolver is not None else TorchRuntimeResolver()
    try:
        if backend not in RUNTIME_BACKENDS:
            raise RuntimeResolutionError(
                f"未知的后端类型: {backend!r}。支持: {' / '.join(RUNTIME_BACKENDS)}"
            )
        if preference not in DEVICE_PREFERENCES:
            raise RuntimeResolutionError(
                f"未知的设备类型: {preference!r}。支持: {' / '.join(DEVICE_PREFERENCES)}"
            )
        resolution = active_resolver.resolve(
            cast(DevicePreference, preference),
            backend=cast(RuntimeBackend, backend),
        )
    except RuntimeResolutionError as exc:
        raise DeviceNotAvailableError(str(exc)) from exc
    if resolution.fallback_reason is not None:
        logging.getLogger("music_sep").warning("%r", resolution.fallback_reason)
    return resolution


def detect_device(
    preference: str = "auto",
    backend: str = "torch",
    *,
    resolver: RuntimeResolver | None = None,
) -> str:
    """兼容旧字符串 API，返回本次解析得到的实际设备。"""

    return resolve_device(
        preference,
        backend,
        resolver=resolver,
    ).actual


def format_duration(seconds: float) -> str:
    """将秒数格式化为 MM:SS 字符串"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"
