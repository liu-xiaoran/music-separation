import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

from music_sep.adapters.torchaudio_writer import TorchaudioAudioWriter
from music_sep.domain.errors import AudioWriteError


def assert_private_file_mode(path: Path) -> None:
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


class RecordingSaver:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, Path, int, dict[str, Any]]] = []
        self.error: Exception | None = None
        self.payload = b"encoded"

    def __call__(
        self,
        audio: torch.Tensor,
        path: str | Path,
        samplerate: int,
        **kwargs: Any,
    ) -> None:
        output_path = Path(path)
        self.calls.append((audio.detach().clone(), output_path, samplerate, kwargs))
        output_path.write_bytes(b"partial" if self.error is not None else self.payload)
        if self.error is not None:
            raise self.error


def test_wav_write_normalizes_mono_and_uses_pcm16(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)
    output_path = tmp_path / "vocals.wav"

    writer.write(torch.arange(8, dtype=torch.float32), output_path, 44100, "wav", "128k")

    assert output_path.read_bytes() == b"encoded"
    assert len(saver.calls) == 1
    audio, temporary_path, samplerate, kwargs = saver.calls[0]
    assert audio.shape == (1, 8)
    assert audio.device.type == "cpu"
    assert temporary_path != output_path
    assert temporary_path.suffix == ".wav"
    assert samplerate == 44100
    assert kwargs == {
        "bitrate": 320,
        "bits_per_sample": 16,
        "as_float": False,
    }
    assert_private_file_mode(output_path)
    assert list(tmp_path.iterdir()) == [output_path]


def test_mp3_write_parses_bitrate_case_insensitively(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)

    writer.write(torch.zeros(2, 8), tmp_path / "vocals.mp3", 48000, "mp3", "192K")

    _, temporary_path, samplerate, kwargs = saver.calls[0]
    assert temporary_path.suffix == ".mp3"
    assert samplerate == 48000
    assert kwargs["bitrate"] == 192


def test_flac_write_keeps_stereo_shape(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)
    audio = torch.randn(2, 16)

    writer.write(audio, tmp_path / "drums.flac", 44100, "flac", "128k")

    saved_audio = saver.calls[0][0]
    torch.testing.assert_close(saved_audio, audio)
    assert saved_audio.shape == (2, 16)


@pytest.mark.parametrize(
    ("fmt", "expected_format", "dtype"),
    [
        ("wav", "WAV", torch.float16),
        ("wav", "WAV", torch.bfloat16),
        ("flac", "FLAC", torch.float16),
        ("flac", "FLAC", torch.bfloat16),
    ],
)
def test_default_writer_encodes_readable_pcm16_audio(
    tmp_path: Path,
    fmt: str,
    expected_format: str,
    dtype: torch.dtype,
) -> None:
    writer = TorchaudioAudioWriter()
    output_path = tmp_path / f"vocals.{fmt}"
    left = torch.linspace(-0.5, 0.5, 800)
    audio = torch.stack((left, -left)).to(dtype)

    writer.write(audio, output_path, 8000, fmt, "ignored")

    info = sf.info(output_path)
    decoded, decoded_samplerate = sf.read(output_path, dtype="float32", always_2d=True)
    assert info.format == expected_format
    assert info.subtype == "PCM_16"
    assert info.samplerate == decoded_samplerate == 8000
    assert info.channels == decoded.shape[1] == 2
    assert info.frames == decoded.shape[0] == 800
    np.testing.assert_allclose(decoded, audio.float().transpose(0, 1).numpy(), atol=2 / 32768)
    assert_private_file_mode(output_path)
    assert list(tmp_path.iterdir()) == [output_path]


def test_default_writer_encodes_decodable_mp3_without_torchaudio_codec(
    tmp_path: Path,
) -> None:
    writer = TorchaudioAudioWriter()
    output_path = tmp_path / "vocals.mp3"
    phase = torch.linspace(0, 20, 800)
    audio = torch.stack((torch.sin(phase), torch.cos(phase))) * 0.2

    writer.write(audio, output_path, 8000, "mp3", "64k")

    decoded, samplerate = sf.read(output_path, dtype="float32", always_2d=True)
    assert output_path.stat().st_size > 100
    assert samplerate == 8000
    assert decoded.shape[0] >= 800
    assert decoded.shape[1] == 2
    assert np.max(np.abs(decoded)) > 0.05
    assert_private_file_mode(output_path)
    assert list(tmp_path.iterdir()) == [output_path]


def test_singleton_batch_dimension_is_removed(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)

    writer.write(torch.randn(1, 2, 16), tmp_path / "bass.wav", 44100, "wav", "128k")

    assert saver.calls[0][0].shape == (2, 16)


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (torch.randn(2, 2, 16), "二维"),
        (torch.tensor([[float("nan")]]), "有限"),
        (torch.tensor([[float("inf")]]), "有限"),
    ],
)
def test_invalid_tensor_is_rejected_before_file_creation(
    tmp_path: Path,
    audio: torch.Tensor,
    message: str,
) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)

    with pytest.raises(AudioWriteError, match=message):
        writer.write(audio, tmp_path / "bad.wav", 44100, "wav", "128k")

    assert saver.calls == []
    assert list(tmp_path.iterdir()) == []


def test_non_tensor_input_is_rejected(tmp_path: Path) -> None:
    writer = TorchaudioAudioWriter(saver=RecordingSaver())

    with pytest.raises(AudioWriteError, match="Tensor"):
        writer.write([0.0], tmp_path / "bad.wav", 44100, "wav", "128k")


def test_non_mp3_formats_ignore_bitrate(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)

    for fmt in ("wav", "flac"):
        writer.validate(fmt, "not-an-mp3-bitrate")
        writer.write(
            torch.zeros(1, 8),
            tmp_path / f"unused.{fmt}",
            8000,
            fmt,
            "not-an-mp3-bitrate",
        )

    assert [call[3]["bitrate"] for call in saver.calls] == [320, 320]


def test_invalid_format_and_mp3_bitrate_are_rejected() -> None:
    writer = TorchaudioAudioWriter(saver=RecordingSaver())

    with pytest.raises(AudioWriteError, match="不支持的音频格式"):
        writer.validate("ogg", "128k")
    with pytest.raises(AudioWriteError, match="MP3 比特率"):
        writer.validate("mp3", "fast")
    with pytest.raises(AudioWriteError, match="8.*320"):
        writer.validate("mp3", "512k")


def test_failed_encode_preserves_existing_destination_and_removes_temp_file(
    tmp_path: Path,
) -> None:
    saver = RecordingSaver()
    saver.error = RuntimeError("codec failed")
    writer = TorchaudioAudioWriter(saver=saver)
    output_path = tmp_path / "vocals.wav"
    output_path.write_bytes(b"old")

    with pytest.raises(AudioWriteError, match="音频编码失败") as exc_info:
        writer.write(torch.zeros(2, 8), output_path, 44100, "wav", "128k")

    assert exc_info.value.__cause__ is saver.error
    assert output_path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [output_path]


def test_failed_replace_preserves_existing_destination_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = TorchaudioAudioWriter(saver=RecordingSaver())
    output_path = tmp_path / "vocals.wav"
    output_path.write_bytes(b"old")

    def fail_replace(path: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(AudioWriteError, match="发布音频文件失败") as exc_info:
        writer.write(torch.zeros(2, 8), output_path, 44100, "wav", "128k")

    assert isinstance(exc_info.value.__cause__, OSError)
    assert output_path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [output_path]


def test_successful_encode_atomically_replaces_existing_destination(tmp_path: Path) -> None:
    saver = RecordingSaver()
    writer = TorchaudioAudioWriter(saver=saver)
    output_path = tmp_path / "vocals.wav"
    output_path.write_bytes(b"old")

    writer.write(torch.zeros(2, 8), output_path, 44100, "wav", "128k")

    assert output_path.read_bytes() == b"encoded"
    assert_private_file_mode(output_path)
    assert list(tmp_path.iterdir()) == [output_path]
