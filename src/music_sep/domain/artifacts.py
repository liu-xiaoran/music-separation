from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeparationOptions:
    """Demucs 分离所需的不可变运行参数。"""

    model: str
    device: str
    shifts: int
    overlap: float
    two_stems: str | None = None
    stems: tuple[str, ...] | None = None
