from __future__ import annotations

import os
import re
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from music_sep.domain.errors import AudioWriteError

AudioSaver = Callable[..., None]
_VALID_FORMATS = frozenset({"wav", "flac", "mp3"})
_BITRATE_PATTERN = re.compile(r"^([1-9]\d{0,2})[kK]$")


class TorchaudioAudioWriter:
    """通过 SoundFile/lameenc 编码并原子发布单条音轨。"""

    def __init__(self, *, saver: AudioSaver | None = None) -> None:
        self._saver = saver

    @staticmethod
    def _parse_bitrate(bitrate: str) -> int:
        if not isinstance(bitrate, str):
            raise AudioWriteError("MP3 比特率必须是形如 128k 的字符串")
        match = _BITRATE_PATTERN.fullmatch(bitrate)
        if match is None:
            raise AudioWriteError("MP3 比特率必须是形如 128k 的字符串")
        value = int(match.group(1))
        if not 8 <= value <= 320:
            raise AudioWriteError("MP3 比特率必须在 8k 到 320k 之间")
        return value

    def validate(self, fmt: str, bitrate: str) -> None:
        if not isinstance(fmt, str) or fmt.lower() not in _VALID_FORMATS:
            raise AudioWriteError(
                f"不支持的音频格式: {fmt}。支持: {', '.join(sorted(_VALID_FORMATS))}"
            )
        if fmt.lower() == "mp3":
            self._parse_bitrate(bitrate)

    @staticmethod
    def _prepare_audio(audio: object) -> Any:
        import torch

        if not isinstance(audio, torch.Tensor):
            raise AudioWriteError("音频数据必须是 torch.Tensor")

        prepared = audio.detach()
        if prepared.ndim == 1:
            prepared = prepared.unsqueeze(0)
        while prepared.ndim > 2 and prepared.shape[0] == 1:
            prepared = prepared.squeeze(0)
        if prepared.ndim != 2 or prepared.shape[0] == 0 or prepared.shape[1] == 0:
            raise AudioWriteError(
                f"音频 Tensor 必须为非空二维 (channels, samples)，当前形状: {tuple(prepared.shape)}"
            )
        if prepared.is_complex():
            raise AudioWriteError("音频 Tensor 不支持复数类型")
        if prepared.dtype not in (torch.float32, torch.float64):
            prepared = prepared.float()
        prepared = prepared.cpu().clone()
        if not torch.isfinite(prepared).all():
            raise AudioWriteError("音频 Tensor 必须只包含有限数值")
        return prepared

    @staticmethod
    def _default_saver(
        audio: Any,
        path: Path,
        samplerate: int,
        *,
        bitrate: int,
        bits_per_sample: int,
        as_float: bool,
    ) -> None:
        try:
            from demucs.audio import encode_mp3, prevent_clip
        except ImportError as exc:
            raise AudioWriteError(f"无法导入 Demucs 音频编码后端: {exc}") from exc

        prepared = prevent_clip(audio, mode="rescale")
        suffix = path.suffix.lower()
        if suffix == ".mp3":
            encode_mp3(
                prepared,
                path,
                samplerate=samplerate,
                bitrate=bitrate,
                quality=2,
                verbose=True,
            )
            return

        try:
            soundfile = import_module("soundfile")
        except ImportError as exc:
            raise AudioWriteError(f"无法导入 SoundFile 音频编码后端: {exc}") from exc

        output_format = "WAV" if suffix == ".wav" else "FLAC"
        subtype = "FLOAT" if as_float else f"PCM_{bits_per_sample}"
        soundfile.write(
            str(path),
            prepared.transpose(0, 1).numpy(),
            samplerate,
            format=output_format,
            subtype=subtype,
        )

    def _get_saver(self) -> AudioSaver:
        return self._saver if self._saver is not None else self._default_saver

    def write(
        self,
        audio: object,
        output_path: Path,
        samplerate: int,
        fmt: str,
        bitrate: str,
    ) -> None:
        self.validate(fmt, bitrate)
        if isinstance(samplerate, bool) or not isinstance(samplerate, int) or samplerate <= 0:
            raise AudioWriteError(f"采样率必须是正整数，当前值: {samplerate}")

        normalized_format = fmt.lower()
        if output_path.suffix.lower() != f".{normalized_format}":
            raise AudioWriteError(
                f"输出路径扩展名 {output_path.suffix} 与格式 {normalized_format} 不一致"
            )
        prepared = self._prepare_audio(audio)
        bitrate_value = self._parse_bitrate(bitrate) if normalized_format == "mp3" else 320

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_dir = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_path.stem}.",
                    suffix=".tmp",
                    dir=output_path.parent,
                )
            )
        except OSError as exc:
            raise AudioWriteError(f"无法创建音频临时目录: {exc}") from exc

        temporary_path = temporary_dir / output_path.name
        try:
            saver = self._get_saver()
            try:
                saver(
                    prepared,
                    temporary_path,
                    samplerate,
                    bitrate=bitrate_value,
                    bits_per_sample=16,
                    as_float=False,
                )
            except AudioWriteError:
                raise
            except Exception as exc:
                raise AudioWriteError(f"音频编码失败: {exc}") from exc

            if os.name != "nt":
                try:
                    temporary_path.chmod(0o600)
                except OSError as exc:
                    raise AudioWriteError(f"设置音频文件权限失败: {exc}") from exc

            try:
                temporary_path.replace(output_path)
            except OSError as exc:
                raise AudioWriteError(f"发布音频文件失败: {exc}") from exc
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                temporary_dir.rmdir()
            except OSError:
                pass
