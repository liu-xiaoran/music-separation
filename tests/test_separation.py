import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from music_sep.config import SeparationConfig
from music_sep.separation import SeparationEngine
from music_sep.outputs import OutputPaths
from music_sep.exceptions import AudioProcessingError, ModelNotFoundError


@pytest.fixture
def sep_config():
    return SeparationConfig(model="htdemucs", device="cpu", shifts=1, overlap=0.25)


@pytest.fixture
def output_paths(tmp_path):
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    return OutputPaths(
        base_dir=tmp_path,
        stems_dir=stems_dir,
        lyrics_path=tmp_path / "lyrics.srt",
        viz_dir=tmp_path / "viz",
    )


class TestSeparationEngine:
    def test_init(self, sep_config):
        engine = SeparationEngine(sep_config)
        assert engine._separator is None

    @patch("music_sep.separation.SeparationEngine._get_separator")
    def test_separate_returns_stems(self, mock_get_sep, sep_config, output_paths, sample_audio):
        # Mock separator
        mock_sep = MagicMock()
        mock_sep.samplerate = 44100

        import numpy as np
        import torch
        # 创建假 tensor（2声道，1000采样点）
        fake_tensor = torch.randn(2, 1000)
        mock_sep.separate_audio_file.return_value = (
            fake_tensor,
            {"vocals": fake_tensor, "drums": fake_tensor, "bass": fake_tensor, "other": fake_tensor},
        )
        mock_get_sep.return_value = mock_sep

        engine = SeparationEngine(sep_config)
        results = engine.separate(sample_audio, output_paths)

        assert "vocals" in results
        assert "drums" in results
        assert len(results) == 4

    @patch("music_sep.separation.SeparationEngine._get_separator")
    def test_separate_with_stems_filter(self, mock_get_sep, sep_config, output_paths, sample_audio):
        config = SeparationConfig(model="htdemucs", device="cpu", stems=["vocals", "drums"])

        mock_sep = MagicMock()
        mock_sep.samplerate = 44100

        import torch
        fake_tensor = torch.randn(2, 1000)
        mock_sep.separate_audio_file.return_value = (
            fake_tensor,
            {"vocals": fake_tensor, "drums": fake_tensor, "bass": fake_tensor, "other": fake_tensor},
        )
        mock_get_sep.return_value = mock_sep

        engine = SeparationEngine(config)
        results = engine.separate(sample_audio, output_paths)

        assert set(results.keys()) == {"vocals", "drums"}
