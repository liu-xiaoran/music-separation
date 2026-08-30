from __future__ import annotations

import logging
from pathlib import Path

from music_sep.adapters.demucs_separator import DemucsSeparator
from music_sep.adapters.torchaudio_writer import TorchaudioAudioWriter
from music_sep.config import SeparationConfig
from music_sep.domain.artifacts import SeparationOptions
from music_sep.domain.errors import AudioWriteError, DemucsModelError, SeparationAdapterError
from music_sep.exceptions import AudioProcessingError, ModelNotFoundError
from music_sep.outputs import OutputPaths, stem_filename
from music_sep.ports.audio_writer import AudioWriter
from music_sep.ports.separator import Separator

logger = logging.getLogger("music_sep")


class SeparationEngine:
    """保持既有 API 的音乐分离 facade。"""

    def __init__(
        self,
        config: SeparationConfig,
        *,
        separator: Separator | None = None,
        audio_writer: AudioWriter | None = None,
    ) -> None:
        self._config = config
        options = SeparationOptions(
            model=config.model,
            device=config.device,
            shifts=config.shifts,
            overlap=config.overlap,
            two_stems=config.two_stems,
            stems=tuple(config.stems) if config.stems is not None else None,
        )
        self._separator = separator if separator is not None else DemucsSeparator(options)
        self._audio_writer = audio_writer if audio_writer is not None else TorchaudioAudioWriter()

    def validate_output(self, output_format: str = "wav", bitrate: str = "128k") -> None:
        """在模型推理或输出目录变更前校验音频输出参数。"""
        try:
            self._audio_writer.validate(output_format, bitrate)
        except AudioWriteError as exc:
            raise AudioProcessingError(f"输出参数无效: {exc}") from exc
        except Exception as exc:
            raise AudioProcessingError(f"输出参数校验失败: {exc}") from exc

    def separate(
        self,
        input_file: Path,
        output_paths: OutputPaths,
        output_format: str = "wav",
        bitrate: str = "128k",
    ) -> dict[str, Path]:
        """执行音乐分离并保存音轨文件。

        Returns:
            `{stem_name: output_file_path}` 字典。
        """
        self.validate_output(output_format, bitrate)

        try:
            separated_audio = self._separator.separate(input_file)
        except DemucsModelError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        except SeparationAdapterError as exc:
            raise AudioProcessingError(str(exc)) from exc
        except ModelNotFoundError:
            raise
        except AudioProcessingError:
            raise
        except Exception as exc:
            raise AudioProcessingError(f"分离失败: {exc}") from exc

        if not separated_audio.sources:
            raise AudioProcessingError("分离结果为空，未生成任何音轨")

        normalized_format = output_format.lower()
        results: dict[str, Path] = {}
        for stem_name, audio in separated_audio.sources.items():
            output_file = output_paths.stems_dir / stem_filename(
                stem_name,
                normalized_format,
            )
            try:
                self._audio_writer.write(
                    audio,
                    output_file,
                    separated_audio.samplerate,
                    normalized_format,
                    bitrate,
                )
            except AudioWriteError as exc:
                raise AudioProcessingError(f"保存音轨 {stem_name} 失败: {exc}") from exc
            except Exception as exc:
                raise AudioProcessingError(f"保存音轨 {stem_name} 失败: {exc}") from exc
            results[stem_name] = output_file
            logger.info("已保存: %s", output_file.name)

        logger.info("分离完成，共 %d 轨", len(results))
        return results

    def get_model_info(self) -> dict[str, object]:
        """返回当前模型信息。"""
        try:
            return dict(self._separator.get_model_info())
        except DemucsModelError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        except ModelNotFoundError:
            raise
        except Exception as exc:
            raise ModelNotFoundError(f"模型信息读取失败: {exc}") from exc
