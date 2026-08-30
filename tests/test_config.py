import pytest
from pathlib import Path

from music_sep.config import (
    AppConfig,
    SeparationConfig,
    LyricsConfig,
    VisualizationConfig,
    OutputConfig,
    merge_config,
    validate_config,
    load_toml_config,
)
from music_sep.exceptions import ConfigurationError


class TestDefaultConfig:
    def test_default_app_config(self):
        config = AppConfig()
        assert config.separation.model == "htdemucs"
        assert config.separation.device == "auto"
        assert config.separation.shifts == 1
        assert config.separation.overlap == 0.25
        assert config.separation.two_stems is None
        assert config.separation.stems is None

        assert config.lyrics.enabled is False
        assert config.lyrics.whisper_model == "medium"
        assert config.lyrics.whisper_device == "auto"
        assert config.lyrics.language is None
        assert config.lyrics.output_format == "srt"

        assert config.visualization.enabled is False
        assert config.visualization.types == ["waveform", "spectrogram", "mel"]

        assert config.output.output_dir == Path("demo")
        assert config.output.format == "wav"
        assert config.output.bitrate == "128k"
        assert config.output.overwrite is False


class TestMergeConfig:
    def test_cli_overrides_defaults(self):
        config = merge_config(model="htdemucs_ft", device="cpu", shifts=3)
        assert config.separation.model == "htdemucs_ft"
        assert config.separation.device == "cpu"
        assert config.separation.shifts == 3
        # 其他保持默认
        assert config.separation.overlap == 0.25

    def test_toml_overrides_defaults(self, toml_config_file):
        config = merge_config(toml_path=toml_config_file)
        assert config.separation.model == "htdemucs_ft"
        assert config.separation.device == "cpu"
        assert config.separation.shifts == 2
        assert config.lyrics.enabled is True
        assert config.lyrics.whisper_model == "small"
        assert config.lyrics.language == "zh"
        assert config.visualization.enabled is True
        assert config.output.format == "flac"

    def test_cli_overrides_toml(self, toml_config_file):
        config = merge_config(
            toml_path=toml_config_file,
            model="mdx",
            shifts=5,
        )
        assert config.separation.model == "mdx"
        assert config.separation.shifts == 5
        # TOML 的 device 仍然生效
        assert config.separation.device == "cpu"

    def test_bool_three_state(self):
        # None (未指定) → 回退到默认 False
        config = merge_config(lyrics_enabled=None)
        assert config.lyrics.enabled is False

        # True (显式开启)
        config = merge_config(lyrics_enabled=True)
        assert config.lyrics.enabled is True

        # False (显式关闭)
        config = merge_config(lyrics_enabled=False)
        assert config.lyrics.enabled is False

    def test_bool_three_state_with_toml(self, toml_config_file):
        # TOML 中 enabled=true, CLI 显式关闭
        config = merge_config(
            toml_path=toml_config_file,
            lyrics_enabled=False,
        )
        assert config.lyrics.enabled is False


class TestValidateConfig:
    def test_invalid_model(self):
        config = AppConfig(separation=SeparationConfig(model="nonexistent"))
        with pytest.raises(ConfigurationError, match="未知的 Demucs 模型"):
            validate_config(config)

    def test_invalid_whisper_model(self):
        config = AppConfig(lyrics=LyricsConfig(whisper_model="nonexistent"))
        with pytest.raises(ConfigurationError, match="未知的 Whisper 模型"):
            validate_config(config)

    def test_invalid_lyrics_format(self):
        config = AppConfig(lyrics=LyricsConfig(output_format="pdf"))
        with pytest.raises(ConfigurationError, match="不支持的歌词格式"):
            validate_config(config)

    def test_invalid_output_format(self):
        config = AppConfig(output=OutputConfig(format="ogg"))
        with pytest.raises(ConfigurationError, match="不支持的输出格式"):
            validate_config(config)

    def test_invalid_viz_type(self):
        config = AppConfig(visualization=VisualizationConfig(types=["waveform", "nonexistent"]))
        with pytest.raises(ConfigurationError, match="未知的可视化类型"):
            validate_config(config)

    def test_two_stems_and_stems_mutual_exclusion(self):
        config = AppConfig(
            separation=SeparationConfig(two_stems="vocals", stems=["vocals", "drums"])
        )
        with pytest.raises(ConfigurationError, match="--two-stems 和 --stems"):
            validate_config(config)

    def test_invalid_overlap(self):
        config = AppConfig(separation=SeparationConfig(overlap=1.5))
        with pytest.raises(ConfigurationError, match="overlap"):
            validate_config(config)

    def test_invalid_shifts(self):
        config = AppConfig(separation=SeparationConfig(shifts=0))
        with pytest.raises(ConfigurationError, match="shifts"):
            validate_config(config)


class TestLoadTomlConfig:
    def test_nonexistent_file(self):
        with pytest.raises(ConfigurationError, match="配置文件不存在"):
            load_toml_config(Path("/nonexistent/config.toml"))

    def test_invalid_toml(self, tmp_path):
        bad_file = tmp_path / "bad.toml"
        bad_file.write_text("invalid [ toml")
        with pytest.raises(ConfigurationError, match="解析失败"):
            load_toml_config(bad_file)
