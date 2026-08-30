from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AudioWriter(Protocol):
    """音频文件编码端口。"""

    def validate(self, fmt: str, bitrate: str) -> None:
        """在推理开始前校验输出参数。"""

    def write(
        self,
        audio: object,
        output_path: Path,
        samplerate: int,
        fmt: str,
        bitrate: str,
    ) -> None:
        """将单条音轨原子写入目标路径。"""
