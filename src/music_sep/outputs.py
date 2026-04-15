from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from music_sep.exceptions import MusicSepError


@dataclass
class OutputPaths:
    """单首歌曲的所有输出路径"""
    base_dir: Path          # demo/<song_name>/
    stems_dir: Path         # demo/<song_name>/stems/
    lyrics_path: Path       # demo/<song_name>/lyrics.<format>
    viz_dir: Path           # demo/<song_name>/visualizations/


class OutputExistsError(MusicSepError):
    """输出目录已存在且未开启覆盖"""


def resolve_output_paths(
    input_file: Path,
    output_root: Path,
    lyrics_format: str = "srt",
    overwrite: bool = False,
) -> OutputPaths:
    """根据输入文件和输出根目录解析所有输出路径

    注意：此函数仅计算路径，不创建任何目录。
    目录创建由 ensure_output_dirs() 负责。

    Args:
        input_file: 输入音频文件路径
        output_root: 输出根目录（如 demo/）
        lyrics_format: 歌词文件格式扩展名
        overwrite: 是否覆盖已有输出

    Returns:
        OutputPaths 实例

    Raises:
        OutputExistsError: 输出目录已存在且 overwrite=False
    """
    song_name = input_file.stem
    base_dir = output_root / song_name

    if base_dir.exists() and not overwrite:
        raise OutputExistsError(
            f"输出目录已存在: {base_dir}。使用 --overwrite 覆盖。"
        )

    return OutputPaths(
        base_dir=base_dir,
        stems_dir=base_dir / "stems",
        lyrics_path=base_dir / f"lyrics.{lyrics_format}",
        viz_dir=base_dir / "visualizations",
    )


def ensure_output_dirs(paths: OutputPaths, overwrite: bool = False) -> None:
    """实际创建输出所需的目录结构

    Args:
        paths: 输出路径配置
        overwrite: 是否覆盖已有目录

    Raises:
        OutputExistsError: base_dir 是文件而非目录
    """
    if paths.base_dir.exists() and paths.base_dir.is_file():
        raise OutputExistsError(
            f"输出路径已被文件占用: {paths.base_dir}"
        )

    if overwrite and paths.base_dir.exists():
        shutil.rmtree(paths.base_dir)

    paths.stems_dir.mkdir(parents=True, exist_ok=True)
    paths.viz_dir.mkdir(parents=True, exist_ok=True)


def stem_filename(stem_name: str, fmt: str) -> str:
    """生成音轨文件名"""
    return f"{stem_name}.{fmt}"


def viz_filename(viz_type: str, stem_name: Optional[str] = None) -> str:
    """生成可视化文件名

    Args:
        viz_type: 可视化类型（waveform/spectrogram/mel）
        stem_name: 音轨名称，None 表示原始音频

    Returns:
        文件名字符串
    """
    if stem_name:
        return f"{viz_type}_{stem_name}.png"
    return f"{viz_type}_original.png"
