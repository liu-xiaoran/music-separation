from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from music_sep.config import SeparationConfig
from music_sep.exceptions import AudioProcessingError, ModelNotFoundError
from music_sep.outputs import OutputPaths, stem_filename

logger = logging.getLogger("music_sep")


class SeparationEngine:
    """封装 demucs.api.Separator"""

    def __init__(self, config: SeparationConfig):
        self._config = config
        self._separator = None

    def _get_separator(self):
        """懒加载初始化 Separator 实例"""
        if self._separator is None:
            try:
                from demucs.api import Separator
            except ImportError as e:
                raise ModelNotFoundError(f"无法导入 demucs: {e}")

            kwargs = dict(
                model=self._config.model,
                device=self._config.device,
                shifts=self._config.shifts,
                overlap=self._config.overlap,
                progress=True,
            )
            if self._config.two_stems:
                kwargs["two_stems"] = self._config.two_stems

            try:
                logger.info(f"加载 Demucs 模型: {self._config.model} (设备: {self._config.device})")
                self._separator = Separator(**kwargs)
            except torch.cuda.OutOfMemoryError:
                raise AudioProcessingError(
                    "GPU 显存不足。尝试使用 --device cpu 或减小 --shifts。"
                )
            except Exception as e:
                raise ModelNotFoundError(f"模型加载失败: {e}")

        return self._separator

    def separate(
        self,
        input_file: Path,
        output_paths: OutputPaths,
        output_format: str = "wav",
        bitrate: str = "128k",
    ) -> dict[str, Path]:
        """执行音乐分离并保存音轨文件

        Returns:
            {stem_name: output_file_path} 字典

        Raises:
            AudioProcessingError: 分离过程中出错
        """
        separator = self._get_separator()

        try:
            logger.info(f"开始分离: {input_file.name}")
            _, separated = separator.separate_audio_file(str(input_file))
        except torch.cuda.OutOfMemoryError:
            raise AudioProcessingError(
                "GPU 显存不足。尝试使用 --device cpu 或减小 --shifts。"
            )
        except Exception as e:
            raise AudioProcessingError(f"分离失败: {e}")

        # 根据 stems 配置过滤
        stems_to_save = separated
        if self._config.stems:
            stems_to_save = {
                name: tensor
                for name, tensor in separated.items()
                if name in self._config.stems
            }

        # 保存音轨文件
        results: dict[str, Path] = {}
        samplerate = separator.samplerate or 44100

        for stem_name, tensor in stems_to_save.items():
            filename = stem_filename(stem_name, output_format)
            output_file = output_paths.stems_dir / filename
            try:
                self._save_audio(tensor, output_file, samplerate, output_format, bitrate)
            except Exception as e:
                raise AudioProcessingError(f"保存音轨 {stem_name} 失败: {e}")
            results[stem_name] = output_file
            logger.info(f"已保存: {output_file.name}")

        logger.info(f"分离完成，共 {len(results)} 轨")
        return results

    def _save_audio(
        self,
        tensor,
        output_path: Path,
        samplerate: int,
        fmt: str,
        bitrate: str = "128k",
    ) -> None:
        """保存音频 tensor 到文件

        Args:
            tensor: PyTorch tensor (channels, samples)
            output_path: 输出路径
            samplerate: 采样率
            fmt: 输出格式 (wav/mp3/flac)
            bitrate: MP3 比特率
        """
        import torchaudio

        # 确保是 2D tensor (channels, samples)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 2:
            tensor = tensor.squeeze()

        if fmt == "mp3":
            # 使用 demucs 内置的 MP3 编码器
            try:
                from demucs.audio import encode_mp3
                import numpy as np

                # 转为 numpy int16 格式
                wav_np = tensor.cpu().numpy()
                # 确保是 2D (channels, samples)
                if wav_np.ndim == 1:
                    wav_np = wav_np[np.newaxis, :]
                # 转为 int16
                wav_int16 = (wav_np * 32767).clip(-32768, 32767).astype(np.int16)
                # 解析比特率数字（如 "128k" -> 128）
                bitrate_num = int(bitrate.rstrip("kK"))
                encode_mp3(wav_int16, str(output_path), samplerate, bitrate=bitrate_num)
            except ImportError:
                # 回退到 torchaudio（需ffmpeg后端）
                torchaudio.save(
                    str(output_path),
                    tensor.cpu(),
                    samplerate,
                    format="mp3",
                )
        elif fmt == "flac":
            torchaudio.save(
                str(output_path),
                tensor.cpu(),
                samplerate,
                format="flac",
            )
        else:  # wav
            torchaudio.save(
                str(output_path),
                tensor.cpu(),
                samplerate,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16,
            )

    def get_model_info(self) -> dict:
        """返回当前模型信息"""
        separator = self._get_separator()
        return {
            "model": self._config.model,
            "samplerate": separator.samplerate,
        }
