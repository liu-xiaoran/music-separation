from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from music_sep.config import SeparationConfig
from music_sep.domain.artifacts import SeparationOptions
from music_sep.domain.errors import AudioWriteError, DemucsModelError, SeparationAdapterError
from music_sep.exceptions import AudioProcessingError, ModelNotFoundError
from music_sep.outputs import OutputPaths
from music_sep.ports.separator import SeparatedAudio
from music_sep.separation import SeparationEngine


@pytest.fixture
def output_paths(tmp_path: Path) -> OutputPaths:
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    return OutputPaths(
        base_dir=tmp_path,
        stems_dir=stems_dir,
        lyrics_path=tmp_path / "lyrics.srt",
        viz_dir=tmp_path / "viz",
    )


class FakeSeparator:
    def __init__(self, sources: dict[str, torch.Tensor] | None = None) -> None:
        self.sources = (
            sources
            if sources is not None
            else {
                "vocals": torch.ones(2, 8),
                "drums": torch.zeros(2, 8),
            }
        )
        self.separate_calls: list[Path] = []
        self.info_calls = 0
        self.error: Exception | None = None

    def separate(self, input_file: Path) -> SeparatedAudio:
        self.separate_calls.append(input_file)
        if self.error is not None:
            raise self.error
        return SeparatedAudio(samplerate=44100, sources=self.sources)

    def get_model_info(self) -> dict[str, object]:
        self.info_calls += 1
        if self.error is not None:
            raise self.error
        return {
            "model": "fake",
            "samplerate": 44100,
            "sources": list(self.sources),
        }


class RecordingWriter:
    def __init__(self) -> None:
        self.validate_calls: list[tuple[str, str]] = []
        self.write_calls: list[tuple[torch.Tensor, Path, int, str, str]] = []
        self.validation_error: Exception | None = None
        self.write_error: Exception | None = None

    def validate(self, fmt: str, bitrate: str) -> None:
        self.validate_calls.append((fmt, bitrate))
        if self.validation_error is not None:
            raise self.validation_error

    def write(
        self,
        audio: object,
        output_path: Path,
        samplerate: int,
        fmt: str,
        bitrate: str,
    ) -> None:
        if self.write_error is not None:
            raise self.write_error
        assert isinstance(audio, torch.Tensor)
        self.write_calls.append((audio, output_path, samplerate, fmt, bitrate))
        output_path.write_bytes(b"audio")


def make_engine(
    separator: FakeSeparator,
    writer: RecordingWriter,
) -> SeparationEngine:
    return SeparationEngine(
        SeparationConfig(model="htdemucs", device="cpu"),
        separator=separator,
        audio_writer=writer,
    )


@patch("music_sep.separation.TorchaudioAudioWriter")
@patch("music_sep.separation.DemucsSeparator")
def test_legacy_constructor_wires_config_to_default_adapters(
    mock_separator: MagicMock,
    mock_writer: MagicMock,
) -> None:
    separator_factory = mock_separator
    writer_factory = mock_writer
    config = SeparationConfig(
        model="mdx_extra",
        device="mps",
        shifts=3,
        overlap=0.5,
        two_stems=None,
        stems=["vocals", "drums"],
    )

    engine = SeparationEngine(config)
    engine.get_model_info()

    separator_factory.assert_called_once_with(
        SeparationOptions(
            model="mdx_extra",
            device="mps",
            shifts=3,
            overlap=0.5,
            two_stems=None,
            stems=("vocals", "drums"),
        )
    )
    writer_factory.assert_called_once_with()
    separator_factory.return_value.get_model_info.assert_called_once_with()


def test_separate_writes_each_returned_stem(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator()
    writer = RecordingWriter()
    engine = make_engine(separator, writer)
    input_file = tmp_path / "input.wav"

    result = engine.separate(input_file, output_paths, output_format="flac", bitrate="192k")

    assert separator.separate_calls == [input_file]
    assert writer.validate_calls == [("flac", "192k")]
    assert set(result) == {"vocals", "drums"}
    assert result["vocals"] == output_paths.stems_dir / "vocals.flac"
    assert result["drums"] == output_paths.stems_dir / "drums.flac"
    assert [call[1].name for call in writer.write_calls] == ["vocals.flac", "drums.flac"]
    assert all(call[2:] == (44100, "flac", "192k") for call in writer.write_calls)


def test_writer_options_are_validated_before_inference(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator()
    writer = RecordingWriter()
    writer.validation_error = AudioWriteError("非法比特率")
    engine = make_engine(separator, writer)

    with pytest.raises(AudioProcessingError, match="非法比特率"):
        engine.separate(
            tmp_path / "input.wav",
            output_paths,
            output_format="mp3",
            bitrate="bad",
        )

    assert separator.separate_calls == []


def test_empty_separation_result_is_rejected_without_writes(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator({})
    writer = RecordingWriter()
    engine = make_engine(separator, writer)

    with pytest.raises(AudioProcessingError, match="分离结果为空"):
        engine.separate(tmp_path / "input.wav", output_paths)

    assert writer.write_calls == []


def test_model_error_preserves_public_exception(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator()
    separator.error = DemucsModelError("下载失败")
    engine = make_engine(separator, RecordingWriter())

    with pytest.raises(ModelNotFoundError, match="下载失败") as exc_info:
        engine.separate(tmp_path / "input.wav", output_paths)

    assert isinstance(exc_info.value.__cause__, DemucsModelError)


def test_separation_error_preserves_public_exception(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator()
    separator.error = SeparationAdapterError("推理失败")
    engine = make_engine(separator, RecordingWriter())

    with pytest.raises(AudioProcessingError, match="推理失败") as exc_info:
        engine.separate(tmp_path / "input.wav", output_paths)

    assert isinstance(exc_info.value.__cause__, SeparationAdapterError)


def test_writer_error_identifies_stem(
    output_paths: OutputPaths,
    tmp_path: Path,
) -> None:
    separator = FakeSeparator({"vocals": torch.ones(2, 8)})
    writer = RecordingWriter()
    writer.write_error = AudioWriteError("编码失败")
    engine = make_engine(separator, writer)

    with pytest.raises(AudioProcessingError, match="保存音轨 vocals 失败") as exc_info:
        engine.separate(tmp_path / "input.wav", output_paths)

    assert isinstance(exc_info.value.__cause__, AudioWriteError)


def test_get_model_info_delegates_and_returns_copy() -> None:
    separator = FakeSeparator()
    engine = make_engine(separator, RecordingWriter())

    info = engine.get_model_info()
    info["model"] = "changed"

    assert separator.info_calls == 1
    assert engine.get_model_info()["model"] == "fake"
