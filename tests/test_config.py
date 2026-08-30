import logging
from dataclasses import replace
from pathlib import Path

import pytest

from music_sep.config import (
    AppConfig,
    SeparationConfig,
    LyricsConfig,
    VisualizationConfig,
    OutputConfig,
    load_toml_config,
    merge_config,
    to_validated_config,
    validate_config,
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


def test_cli_valid_values_override_malformed_lower_priority_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[separation]
shifts = "broken"

[output]
output_dir = 123
""".strip(),
        encoding="utf-8",
    )
    cli_output = tmp_path / "cli-output"

    config = merge_config(
        toml_path=config_path,
        shifts=2,
        output_dir=cli_output,
    )

    assert config.separation.shifts == 2
    assert config.output.output_dir == cli_output


def test_cli_list_replaces_toml_list_and_is_detached(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[visualization]
types = ["waveform", "spectrogram"]
""".strip(),
        encoding="utf-8",
    )
    cli_types = ["mel"]

    config = merge_config(toml_path=config_path, viz_types=cli_types)
    cli_types.append("waveform")

    assert config.visualization.types == ["mel"]


def test_unknown_toml_sections_and_fields_emit_warnings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[separation]
model = "htdemucs"
future_option = true

[future_section]
enabled = true
""".strip(),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="music_sep"):
        config = merge_config(toml_path=config_path)

    assert config.separation.model == "htdemucs"
    assert "future_option" in caplog.text
    assert "future_section" in caplog.text


def test_unknown_toml_keys_escape_terminal_control_characters(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.toml"
    escaped_control = f"{chr(92)}u001b"
    config_path.write_text(
        (
            f'["future{escaped_control}[2J"]\n'
            "enabled = true\n\n"
            "[separation]\n"
            f'"future{escaped_control}[31m" = true'
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="music_sep"):
        merge_config(toml_path=config_path)

    assert "\x1b" not in caplog.text
    assert "\\x1b" in caplog.text


def test_invalid_toml_values_escape_terminal_control_characters(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    escaped_control = f"{chr(92)}u001b"
    config_path.write_text(
        f'[separation]\nshifts = "{escaped_control}[2J"',
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        merge_config(toml_path=config_path)

    message = str(exc_info.value)
    assert "\x1b" not in message
    assert "\\x1b[2J" in message


def test_known_toml_section_must_be_a_table(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('separation = "invalid"', encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"配置节 \[separation\].*期望表"):
        merge_config(toml_path=config_path)


@pytest.mark.parametrize(
    "config",
    [
        AppConfig(separation=SeparationConfig(shifts=True)),
        AppConfig(separation=SeparationConfig(shifts="1")),  # type: ignore[arg-type]
        AppConfig(separation=SeparationConfig(overlap=True)),
        AppConfig(separation=SeparationConfig(overlap="0.5")),  # type: ignore[arg-type]
        AppConfig(lyrics=LyricsConfig(enabled=1)),  # type: ignore[arg-type]
        AppConfig(lyrics=LyricsConfig(language=1)),  # type: ignore[arg-type]
        AppConfig(visualization=VisualizationConfig(types="mel")),  # type: ignore[arg-type]
        AppConfig(output=OutputConfig(overwrite="false")),  # type: ignore[arg-type]
    ],
)
def test_invalid_types_always_raise_configuration_error(config: AppConfig) -> None:
    with pytest.raises(ConfigurationError):
        validate_config(config)


def test_to_validated_config_returns_immutable_detached_collections() -> None:
    config = AppConfig(
        separation=SeparationConfig(stems=["vocals", "drums"]),
        visualization=VisualizationConfig(types=["mel"]),
    )

    validated = to_validated_config(config)
    config.separation.stems.append("bass")  # type: ignore[union-attr]
    config.visualization.types.append("waveform")

    assert validated.separation_stems == ("vocals", "drums")
    assert validated.visualization_types == ("mel",)


def test_to_validated_config_revalidates_direct_validated_instances() -> None:
    validated = to_validated_config(AppConfig())
    invalid = replace(validated, separation_shifts=True)

    with pytest.raises(ConfigurationError, match="shifts"):
        to_validated_config(invalid)


def test_to_validated_config_recanonicalizes_runtime_mutable_aliases() -> None:
    validated = to_validated_config(AppConfig())
    stems = ["vocals"]
    uncanonical = replace(validated, separation_stems=stems)  # type: ignore[arg-type]

    canonical = to_validated_config(uncanonical)
    stems.append("drums")

    assert canonical.separation_stems == ("vocals",)


@pytest.mark.parametrize(
    ("fmt", "bitrate", "valid"),
    [
        ("wav", "anything", True),
        ("flac", "999k", True),
        ("mp3", "8k", True),
        ("mp3", "320K", True),
        ("mp3", "7k", False),
        ("mp3", "321k", False),
        ("mp3", "fast", False),
    ],
)
def test_bitrate_validation_is_mp3_only(fmt: str, bitrate: str, valid: bool) -> None:
    config = AppConfig(output=OutputConfig(format=fmt, bitrate=bitrate))

    if valid:
        validate_config(config)
    else:
        with pytest.raises(ConfigurationError, match="bitrate|比特率"):
            validate_config(config)
