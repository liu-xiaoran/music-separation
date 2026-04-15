from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from music_sep.config import LyricsConfig
from music_sep.exceptions import TranscriptionError
from music_sep.utils import detect_device

logger = logging.getLogger("music_sep")


@dataclass
class TranscriptionResult:
    """歌词识别结果"""
    segments: list[dict]          # [{"start": float, "end": float, "text": str}]
    language: str                 # 检测到的语言代码
    language_probability: float   # 语言检测置信度


class LyricsTranscriber:
    """封装 faster_whisper.WhisperModel"""

    def __init__(self, config: LyricsConfig):
        self._config = config
        self._device = None
        self._model = None

    def _get_model(self):
        """懒加载初始化 WhisperModel"""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise TranscriptionError(f"无法导入 faster_whisper: {e}")

            # 解析设备（ctranslate2 后端不支持 MPS）
            self._device = detect_device(
                self._config.whisper_device, backend="ctranslate2"
            )
            logger.info(
                f"加载 Whisper 模型: {self._config.whisper_model} "
                f"(设备: {self._device})"
            )

            try:
                # CUDA 使用 float16，CPU 使用 int8
                if self._device == "cuda":
                    self._model = WhisperModel(
                        self._config.whisper_model,
                        device="cuda",
                        compute_type="float16",
                    )
                else:
                    self._model = WhisperModel(
                        self._config.whisper_model,
                        device="cpu",
                        compute_type="int8",
                    )
            except Exception as e:
                raise TranscriptionError(f"Whisper 模型加载失败: {e}")

        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """识别音频中的歌词

        Args:
            audio_path: 音频文件路径（优先使用分离出的人声轨）

        Returns:
            TranscriptionResult

        Raises:
            TranscriptionError: 识别失败
        """
        model = self._get_model()

        try:
            logger.info(f"开始歌词识别: {audio_path.name}")
            segments_iter, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
                language=self._config.language,
            )

            segments = []
            for seg in segments_iter:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })

            result = TranscriptionResult(
                segments=segments,
                language=info.language,
                language_probability=info.language_probability,
            )
            logger.info(
                f"识别完成: {len(segments)} 段, "
                f"语言: {info.language} "
                f"(置信度: {info.language_probability:.2f})"
            )
            return result

        except Exception as e:
            raise TranscriptionError(f"歌词识别失败: {e}")

    def save_lyrics(
        self,
        result: TranscriptionResult,
        output_path: Path,
        fmt: str = "srt",
    ) -> Path:
        """将歌词保存为指定格式

        Args:
            result: 识别结果
            output_path: 输出文件路径
            fmt: 输出格式 (srt/vtt/txt/json)

        Returns:
            保存的文件路径
        """
        writers = {
            "srt": self._write_srt,
            "vtt": self._write_vtt,
            "txt": self._write_txt,
            "json": self._write_json,
        }

        writer = writers.get(fmt)
        if writer is None:
            raise TranscriptionError(f"不支持的歌词格式: {fmt}")

        try:
            writer(result.segments, output_path)
        except OSError as e:
            raise TranscriptionError(f"歌词保存失败: {e}")

        logger.info(f"歌词已保存: {output_path.name}")
        return output_path

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """格式化 SRT 时间戳: HH:MM:SS,mmm"""
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """格式化 VTT 时间戳: HH:MM:SS.mmm"""
        seconds = max(0.0, seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def _write_srt(self, segments: list[dict], path: Path) -> None:
        """写入 SRT 格式"""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_timestamp_srt(seg["start"])
            end = self._format_timestamp_srt(seg["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"])
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_vtt(self, segments: list[dict], path: Path) -> None:
        """写入 WebVTT 格式"""
        lines = ["WEBVTT", ""]
        for seg in segments:
            start = self._format_timestamp_vtt(seg["start"])
            end = self._format_timestamp_vtt(seg["end"])
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"])
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_txt(self, segments: list[dict], path: Path) -> None:
        """写入纯文本格式"""
        lines = [seg["text"] for seg in segments]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_json(self, segments: list[dict], path: Path) -> None:
        """写入 JSON 格式"""
        data = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in segments
        ]
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
