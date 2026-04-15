from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无交互后端

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from music_sep.config import VisualizationConfig
from music_sep.exceptions import VisualizationError
from music_sep.outputs import viz_filename

logger = logging.getLogger("music_sep")


class Visualizer:
    """生成波形图和频谱图"""

    def __init__(self, config: VisualizationConfig):
        self._config = config
        self._figure_dpi = 150
        self._figure_size = (12, 4)

    def generate(
        self,
        audio_path: Path,
        output_dir: Path,
        stem_name: Optional[str] = None,
    ) -> list[Path]:
        """为一个音频文件生成所有配置的可视化

        Args:
            audio_path: 音频文件路径
            output_dir: 输出目录
            stem_name: 音轨名称（None 表示原始音频）

        Returns:
            生成的图片路径列表
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
        except Exception as e:
            raise VisualizationError(f"无法加载音频文件: {e}")

        output_files: list[Path] = []

        for viz_type in self._config.types:
            try:
                filename = viz_filename(viz_type, stem_name)
                output_path = output_dir / filename

                if viz_type == "waveform":
                    self.waveform(y, sr, output_path, title_suffix=stem_name)
                elif viz_type == "spectrogram":
                    self.spectrogram(y, sr, output_path, title_suffix=stem_name)
                elif viz_type == "mel":
                    self.mel_spectrogram(y, sr, output_path, title_suffix=stem_name)
                else:
                    logger.warning(f"未知可视化类型: {viz_type}，跳过")
                    continue

                output_files.append(output_path)
                logger.info(f"已生成: {filename}")

            except Exception as e:
                logger.error(f"生成 {viz_type} 失败: {e}")
                # 单个可视化失败不影响其他
                continue

        return output_files

    def waveform(
        self,
        y: np.ndarray,
        sr: int,
        output_path: Path,
        title_suffix: Optional[str] = None,
    ) -> Path:
        """波形图"""
        fig, ax = plt.subplots(figsize=self._figure_size)
        try:
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#3B82F6")

            title = "Waveform"
            if title_suffix:
                title += f" - {title_suffix}"
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")

            fig.tight_layout()
            fig.savefig(output_path, dpi=self._figure_dpi)
            return output_path
        finally:
            plt.close(fig)

    def spectrogram(
        self,
        y: np.ndarray,
        sr: int,
        output_path: Path,
        title_suffix: Optional[str] = None,
    ) -> Path:
        """STFT 频谱图"""
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        fig, ax = plt.subplots(figsize=self._figure_size)
        try:
            img = librosa.display.specshow(
                S_db, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="viridis"
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            title = "Spectrogram"
            if title_suffix:
                title += f" - {title_suffix}"
            ax.set_title(title)

            fig.tight_layout()
            fig.savefig(output_path, dpi=self._figure_dpi)
            return output_path
        finally:
            plt.close(fig)

    def mel_spectrogram(
        self,
        y: np.ndarray,
        sr: int,
        output_path: Path,
        title_suffix: Optional[str] = None,
    ) -> Path:
        """梅尔频谱图"""
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=self._figure_size)
        try:
            img = librosa.display.specshow(
                S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis"
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

            title = "Mel Spectrogram"
            if title_suffix:
                title += f" - {title_suffix}"
            ax.set_title(title)

            fig.tight_layout()
            fig.savefig(output_path, dpi=self._figure_dpi)
            return output_path
        finally:
            plt.close(fig)
