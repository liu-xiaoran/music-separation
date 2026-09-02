from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from music_sep.config import (
    AppConfig,
    LyricsConfig,
    OutputConfig,
    SeparationConfig,
    VisualizationConfig,
    to_validated_config,
)
from music_sep.domain.config import RuntimeResolution
from music_sep.domain.errors import RuntimeResolutionError
from music_sep.exceptions import ConfigurationError
from music_sep.pipeline import Pipeline


@pytest.fixture
def basic_config():
    return AppConfig(
        separation=SeparationConfig(model="htdemucs", device="cpu"),
        output=OutputConfig(output_dir=Path("demo")),
    )


class RecordingRuntimeResolver:
    def __init__(self, torch_actuals: list[str]) -> None:
        self._torch_actuals = iter(torch_actuals)
        self.calls: list[tuple[str, str]] = []

    def __bool__(self) -> bool:
        return False

    def resolve(self, preference: str, *, backend: str) -> RuntimeResolution:
        self.calls.append((preference, backend))
        actual = next(self._torch_actuals) if backend == "torch" else "cpu"
        reason = (
            "ctranslate2 不支持 MPS，已回退到 CPU"
            if backend == "ctranslate2" and preference == "mps"
            else None
        )
        return RuntimeResolution(
            requested=preference,
            actual=actual,
            backend=backend,
            fallback_reason=reason,
        )


class UnavailableLyricsRuntimeResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def resolve(self, preference: str, *, backend: str) -> RuntimeResolution:
        self.calls.append((preference, backend))
        if backend == "ctranslate2":
            raise RuntimeResolutionError("CTranslate2 CUDA 不可用")
        return RuntimeResolution(
            requested=preference,
            actual="cpu",
            backend="torch",
        )


class TestPipeline:
    def test_dry_run(self, basic_config, sample_audio):
        pipeline = Pipeline(basic_config)
        result = pipeline.run(sample_audio, dry_run=True)
        assert result.input_file == sample_audio
        assert len(result.stems) == 0

    def test_nonexistent_input_raises(self, basic_config):
        pipeline = Pipeline(basic_config)
        with pytest.raises(FileNotFoundError):
            pipeline.run(Path("/nonexistent/audio.mp3"))

    def test_failed_run_clears_previous_device_resolution(
        self,
        basic_config: AppConfig,
        sample_audio: Path,
    ) -> None:
        pipeline = Pipeline(basic_config)
        pipeline.run(sample_audio, dry_run=True)
        assert pipeline.resolved_config is not None

        with pytest.raises(FileNotFoundError):
            pipeline.run(Path("/nonexistent/audio.wav"), dry_run=True)

        assert pipeline.resolved_config is None

    @patch("music_sep.pipeline.SeparationEngine")
    def test_separation_stage(self, MockEngine, basic_config, sample_audio, tmp_path):
        # Mock the separation engine
        mock_engine = MagicMock()
        mock_stems = {
            "vocals": tmp_path / "vocals.wav",
            "drums": tmp_path / "drums.wav",
            "bass": tmp_path / "bass.wav",
            "other": tmp_path / "other.wav",
        }
        # Create fake stem files
        for p in mock_stems.values():
            p.touch()

        mock_engine.separate.return_value = mock_stems
        MockEngine.return_value = mock_engine

        config = AppConfig(
            separation=SeparationConfig(model="htdemucs", device="cpu"),
            output=OutputConfig(output_dir=tmp_path / "demo"),
        )
        pipeline = Pipeline(config)
        result = pipeline.run(sample_audio)

        assert result.stems == mock_stems
        assert len(result.stems) == 4

    def test_invalid_mp3_bitrate_preserves_existing_output_on_overwrite(
        self,
        sample_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_root = tmp_path / "demo"
        existing_output = output_root / sample_audio.stem
        existing_output.mkdir(parents=True)
        marker = existing_output / "keep.txt"
        marker.write_text("old output", encoding="utf-8")
        config = AppConfig(
            separation=SeparationConfig(model="htdemucs", device="cpu"),
            output=OutputConfig(
                output_dir=output_root,
                format="mp3",
                bitrate="invalid",
                overwrite=True,
            ),
        )

        with pytest.raises(ConfigurationError, match="MP3 比特率"):
            Pipeline(config)

        assert marker.read_text(encoding="utf-8") == "old output"

    def test_constructor_eagerly_rejects_invalid_direct_config(self) -> None:
        config = AppConfig(separation=SeparationConfig(shifts=True))

        with pytest.raises(ConfigurationError, match="shifts"):
            Pipeline(config)

    def test_constructor_revalidates_direct_validated_config(self) -> None:
        validated = to_validated_config(AppConfig())
        invalid = replace(validated, separation_shifts=True)

        with pytest.raises(ConfigurationError, match="shifts"):
            Pipeline(invalid)

    @patch("music_sep.pipeline.SeparationEngine")
    def test_runs_resolve_fresh_devices_without_mutating_caller_config(
        self,
        mock_engine: MagicMock,
        sample_audio: Path,
        tmp_path: Path,
    ) -> None:
        config = AppConfig(
            separation=SeparationConfig(device="auto"),
            lyrics=LyricsConfig(enabled=True, whisper_device="mps"),
            output=OutputConfig(output_dir=tmp_path / "demo"),
        )
        original = deepcopy(config)
        resolver = RecordingRuntimeResolver(["mps", "cpu"])
        pipeline = Pipeline(config, runtime_resolver=resolver)

        pipeline.run(sample_audio, dry_run=True)
        pipeline.run(sample_audio, dry_run=True)

        assert config == original
        assert resolver.calls == [
            ("auto", "torch"),
            ("auto", "torch"),
        ]
        resolved_devices = [call.args[0].device for call in mock_engine.call_args_list]
        assert resolved_devices == ["mps", "cpu"]
        assert all(call.args[0] is not config.separation for call in mock_engine.call_args_list)

    @patch("music_sep.pipeline.LyricsTranscriber")
    @patch("music_sep.pipeline.SeparationEngine")
    def test_lyrics_stage_receives_resolved_cpu_context_and_vocals_stem(
        self,
        mock_engine: MagicMock,
        mock_transcriber: MagicMock,
        sample_audio: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        output_root = tmp_path / "demo"
        expected_lyrics_path = output_root / sample_audio.stem / "lyrics.srt"
        vocals_path = tmp_path / "vocals.wav"
        transcription = object()
        mock_engine.return_value.separate.return_value = {"vocals": vocals_path}
        mock_transcriber.return_value.transcribe.return_value = transcription
        mock_transcriber.return_value.save_lyrics.return_value = expected_lyrics_path
        config = AppConfig(
            separation=SeparationConfig(device="auto"),
            lyrics=LyricsConfig(
                enabled=True,
                whisper_device="mps",
                output_format="srt",
            ),
            output=OutputConfig(output_dir=output_root),
        )
        original = deepcopy(config)
        resolver = RecordingRuntimeResolver(["mps"])

        with caplog.at_level("WARNING", logger="music_sep"):
            result = Pipeline(config, runtime_resolver=resolver).run(sample_audio)

        lyrics_config = mock_transcriber.call_args.args[0]
        runtime_resolution = mock_transcriber.call_args.kwargs["runtime_resolution"]
        assert lyrics_config.whisper_device == "cpu"
        assert runtime_resolution == RuntimeResolution(
            requested="mps",
            actual="cpu",
            backend="ctranslate2",
            fallback_reason="ctranslate2 不支持 MPS，已回退到 CPU",
        )
        mock_transcriber.return_value.transcribe.assert_called_once_with(vocals_path)
        mock_transcriber.return_value.save_lyrics.assert_called_once_with(
            transcription,
            expected_lyrics_path,
            fmt="srt",
        )
        assert result.lyrics_path == expected_lyrics_path
        assert config == original
        fallback_records = [
            record for record in caplog.records if "ctranslate2 不支持 MPS" in record.getMessage()
        ]
        assert len(fallback_records) == 1

    @patch("music_sep.pipeline.Visualizer")
    @patch("music_sep.pipeline.LyricsTranscriber")
    @patch("music_sep.pipeline.SeparationEngine")
    def test_unavailable_optional_lyrics_device_does_not_block_later_stages(
        self,
        mock_engine: MagicMock,
        mock_transcriber: MagicMock,
        mock_visualizer: MagicMock,
        sample_audio: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        vocals_path = tmp_path / "vocals.wav"
        stems = {"vocals": vocals_path}
        original_viz = tmp_path / "original_waveform.png"
        vocals_viz = tmp_path / "vocals_waveform.png"
        mock_engine.return_value.separate.return_value = stems
        mock_visualizer.return_value.generate.side_effect = [
            [original_viz],
            [vocals_viz],
        ]
        resolver = UnavailableLyricsRuntimeResolver()
        config = AppConfig(
            separation=SeparationConfig(device="cpu"),
            lyrics=LyricsConfig(enabled=True, whisper_device="cuda"),
            visualization=VisualizationConfig(enabled=True, types=["waveform"]),
            output=OutputConfig(output_dir=tmp_path / "demo"),
        )

        with caplog.at_level("INFO", logger="music_sep"):
            result = Pipeline(config, runtime_resolver=resolver).run(sample_audio)

        assert result.stems == stems
        assert result.lyrics_path is None
        assert result.visualization_paths == [original_viz, vocals_viz]
        mock_engine.return_value.separate.assert_called_once()
        mock_transcriber.assert_not_called()
        mock_visualizer.return_value.generate.assert_has_calls(
            [
                call(
                    sample_audio,
                    result.output_paths.viz_dir,
                    stem_name=None,
                ),
                call(
                    vocals_path,
                    result.output_paths.viz_dir,
                    stem_name="vocals",
                ),
            ]
        )
        assert resolver.calls == [
            ("cpu", "torch"),
            ("cuda", "ctranslate2"),
        ]
        assert "歌词识别失败: CTranslate2 CUDA 不可用" in caplog.text
        assert "跳过歌词识别，继续后续处理" in caplog.text
