import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from music_sep.config import AppConfig, SeparationConfig, OutputConfig
from music_sep.pipeline import Pipeline, PipelineResult
from music_sep.outputs import OutputPaths, OutputExistsError


@pytest.fixture
def basic_config():
    return AppConfig(
        separation=SeparationConfig(model="htdemucs", device="cpu"),
        output=OutputConfig(output_dir=Path("demo")),
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
