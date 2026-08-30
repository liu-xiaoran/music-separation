from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SeparatedAudio:
    """分离后的内存音轨及采样率。"""

    samplerate: int
    sources: Mapping[str, object]


class Separator(Protocol):
    """音源分离端口。"""

    def separate(self, input_file: Path) -> SeparatedAudio:
        """分离输入音频并返回内存音轨。"""

    def get_model_info(self) -> dict[str, object]:
        """返回当前模型的稳定元信息。"""
