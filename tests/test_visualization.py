import pytest
from pathlib import Path

from music_sep.config import VisualizationConfig
from music_sep.visualization import Visualizer


@pytest.fixture
def viz_config():
    return VisualizationConfig(
        enabled=True,
        types=["waveform", "spectrogram", "mel"],
    )


@pytest.fixture
def visualizer(viz_config):
    return Visualizer(viz_config)


class TestVisualizer:
    def test_generate_all_types(self, visualizer, sample_audio, tmp_path):
        output_dir = tmp_path / "viz"
        output_dir.mkdir()

        paths = visualizer.generate(sample_audio, output_dir, stem_name="vocals")

        assert len(paths) == 3
        assert all(p.exists() for p in paths)
        assert all(p.suffix == ".png" for p in paths)
        names = [p.name for p in paths]
        assert "waveform_vocals.png" in names
        assert "spectrogram_vocals.png" in names
        assert "mel_vocals.png" in names

    def test_generate_without_stem_name(self, visualizer, sample_audio, tmp_path):
        output_dir = tmp_path / "viz"
        output_dir.mkdir()

        paths = visualizer.generate(sample_audio, output_dir)

        names = [p.name for p in paths]
        assert "waveform_original.png" in names

    def test_single_type(self, sample_audio, tmp_path):
        config = VisualizationConfig(enabled=True, types=["waveform"])
        viz = Visualizer(config)
        output_dir = tmp_path / "viz"
        output_dir.mkdir()

        paths = viz.generate(sample_audio, output_dir)
        assert len(paths) == 1
        assert "waveform" in paths[0].name

    def test_invalid_audio_raises(self, visualizer, tmp_path):
        fake_audio = tmp_path / "nonexistent.wav"
        with pytest.raises(Exception):
            visualizer.generate(fake_audio, tmp_path)
