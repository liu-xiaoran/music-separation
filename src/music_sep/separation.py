from __future__ import annotations

import logging
from pathlib import Path

import torch

from music_sep.config import SeparationConfig
from music_sep.exceptions import AudioProcessingError, ModelNotFoundError
from music_sep.outputs import OutputPaths, stem_filename

logger = logging.getLogger("music_sep")


class SeparationEngine:
    """封装 demucs 4.0.x 低级 API"""

    def __init__(self, config: SeparationConfig):
        self._config = config
        self._model = None
        self._model_info: dict | None = None

    def _load_model(self):
        """懒加载 demucs 模型"""
        if self._model is not None:
            return

        try:
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels
        except ImportError as e:
            raise ModelNotFoundError(f"无法导入 demucs: {e}")

        try:
            logger.info(f"加载 Demucs 模型: {self._config.model} (设备: {self._config.device})")
            model = get_model(self._config.model)
        except Exception as e:
            raise ModelNotFoundError(f"模型加载失败: {e}")

        # 将模型移到目标设备
        device = self._config.device
        model.to(device)

        # 如果是 BagOfModels 且设置了 two_stems，标记之
        if self._config.two_stems and isinstance(model, BagOfModels):
            model.two_stems = self._config.two_stems

        self._model = model
        self._model_info = {
            "model": self._config.model,
            "samplerate": model.samplerate,
            "sources": list(model.sources),
        }

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
        """
        self._load_model()

        try:
            from demucs.separate import load_track
            from demucs.apply import apply_model
        except ImportError as e:
            raise AudioProcessingError(f"无法导入 demucs: {e}")

        try:
            logger.info(f"开始分离: {input_file.name}")

            # 加载音频
            wav = load_track(
                str(input_file),
                audio_channels=self._model.audio_channels,
                samplerate=self._model.samplerate,
            )

            # 将音频移到目标设备
            device = self._config.device
            ref = wav.mean(0)
            wav = wav.unsqueeze(0).to(device)

            # 应用模型分离
            estimated = apply_model(
                self._model,
                wav,
                shifts=self._config.shifts,
                overlap=self._config.overlap,
                progress=True,
                device=device,
            )

            estimated = estimated[0]  # 去掉 batch 维度

        except torch.cuda.OutOfMemoryError:
            raise AudioProcessingError(
                "GPU 显存不足。尝试使用 --device cpu 或减小 --shifts。"
            )
        except Exception as e:
            raise AudioProcessingError(f"分离失败: {e}")

        # two_stems 模式：仅返回指定轨 + 残差
        sources = self._model.sources
        if self._config.two_stems:
            # 找到目标 stem 的索引
            target_idx = sources.index(self._config.two_stems)
            # 目标音轨
            target_tensor = estimated[target_idx] + ref.to(estimated.device)
            # 残差（其他所有轨的混合）
            other_indices = [i for i in range(len(sources)) if i != target_idx]
            no_target_tensor = sum(
                estimated[i] for i in other_indices
            ) + ref.to(estimated.device)

            separated = {
                self._config.two_stems: target_tensor,
                "no_" + self._config.two_stems: no_target_tensor,
            }
        else:
            separated = {
                name: estimated[i] + ref.to(estimated.device)
                for i, name in enumerate(sources)
            }

        # 根据 stems 配置过滤
        if self._config.stems:
            separated = {
                name: tensor
                for name, tensor in separated.items()
                if name in self._config.stems
            }

        # 保存音轨文件
        results: dict[str, Path] = {}
        samplerate = self._model.samplerate

        for stem_name, tensor in separated.items():
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
        """保存音频 tensor 到文件"""
        import torchaudio

        # 确保是 2D tensor (channels, samples)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 2:
            tensor = tensor.squeeze()

        if fmt == "mp3":
            # 使用 demucs 内置的保存方法
            try:
                from demucs.separate import save_audio
                bitrate_num = int(bitrate.rstrip("kK"))
                save_audio(tensor.cpu(), output_path, samplerate, bitrate=bitrate_num)
                return
            except ImportError:
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
        self._load_model()
        return self._model_info
