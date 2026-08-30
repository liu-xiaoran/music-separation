import logging
import sys
from pathlib import Path

import torch

from music_sep.constants import SUPPORTED_INPUT_EXTENSIONS
from music_sep.exceptions import UnsupportedFormatError, DeviceNotAvailableError


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


def detect_device(preference: str = "auto", backend: str = "torch") -> str:
    """检测可用的计算设备

    Args:
        preference: "cpu", "cuda", "mps", 或 "auto"
        backend: "torch"（Demucs）或 "ctranslate2"（faster-whisper）
            - torch 后端支持: cuda, mps, cpu
            - ctranslate2 后端支持: cuda, cpu（不支持 mps，auto 时 mps 回退 cpu）

    Returns:
        实际可用的设备字符串

    Raises:
        DeviceNotAvailableError: 请求的设备不可用
    """
    if preference == "cpu":
        return "cpu"

    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise DeviceNotAvailableError("CUDA 不可用。请确认已安装 NVIDIA GPU 和 CUDA 版 PyTorch。")

    if preference == "mps":
        if backend == "ctranslate2":
            raise DeviceNotAvailableError(
                "faster-whisper (ctranslate2) 不支持 MPS，请使用 --whisper-device cpu 或 --whisper-device cuda。"
            )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise DeviceNotAvailableError(
            "MPS 不可用。请确认使用的是 Apple Silicon Mac 且已安装正确版本的 PyTorch。"
        )

    if preference != "auto":
        raise DeviceNotAvailableError(
            f"未知的设备类型: {preference}。支持: cpu / cuda / mps / auto"
        )

    # preference == "auto"
    if backend not in ("torch", "ctranslate2"):
        raise DeviceNotAvailableError(f"未知的后端类型: {backend}。支持: torch / ctranslate2")
    if backend == "torch":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    else:  # ctranslate2
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def format_duration(seconds: float) -> str:
    """将秒数格式化为 MM:SS 字符串"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"
