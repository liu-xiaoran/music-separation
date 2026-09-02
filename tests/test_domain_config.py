from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from music_sep.domain.config import (
    RawConfig,
    ResolvedConfig,
    RuntimeResolution,
    ValidatedConfig,
    validate_raw_config,
)
from music_sep.domain.errors import ConfigValidationError, RuntimeResolutionError


def make_raw_config(**overrides: object) -> RawConfig:
    values: dict[str, object] = {
        "separation_model": "htdemucs",
        "separation_device": "auto",
        "separation_shifts": 1,
        "separation_overlap": 0.25,
        "separation_two_stems": None,
        "separation_stems": None,
        "lyrics_enabled": False,
        "whisper_model": "medium",
        "whisper_device": "auto",
        "lyrics_language": None,
        "lyrics_format": "srt",
        "visualization_enabled": False,
        "visualization_types": ("waveform", "spectrogram", "mel"),
        "output_dir": Path("demo"),
        "output_format": "wav",
        "output_bitrate": "128k",
        "output_overwrite": False,
    }
    values.update(overrides)
    return RawConfig(**values)  # type: ignore[arg-type]


def test_validate_raw_config_returns_detached_frozen_snapshot() -> None:
    stems = ["vocals", "drums"]
    visualizations = ["mel"]
    raw = make_raw_config(
        separation_stems=tuple(stems),
        visualization_types=tuple(visualizations),
    )

    validated = validate_raw_config(raw)
    stems.append("bass")
    visualizations.append("waveform")

    assert isinstance(validated, ValidatedConfig)
    assert validated.separation_stems == ("vocals", "drums")
    assert validated.visualization_types == ("mel",)
    with pytest.raises(FrozenInstanceError):
        validated.separation_device = "cpu"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"separation_shifts": True}, "shifts"),
        ({"separation_shifts": 1.5}, "shifts"),
        ({"separation_overlap": True}, "overlap"),
        ({"separation_overlap": float("nan")}, "overlap"),
        ({"separation_overlap": float("inf")}, "overlap"),
        ({"lyrics_enabled": 1}, "lyrics.enabled"),
        ({"visualization_enabled": "false"}, "visualization.enabled"),
        ({"output_overwrite": 0}, "output.overwrite"),
        ({"separation_device": "tpu"}, "separation.device"),
        ({"whisper_device": "tpu"}, "lyrics.whisper_device"),
        ({"separation_stems": ()}, "stems"),
        ({"separation_stems": ("vocals", 1)}, "stems"),
        ({"visualization_types": "mel"}, "visualization.types"),
        ({"output_dir": 123}, "output.output_dir"),
        ({"output_bitrate": 128}, "output.bitrate"),
        ({"output_format": "mp3", "output_bitrate": "7k"}, "output.bitrate"),
    ],
)
def test_invalid_raw_values_raise_controlled_validation_error(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        validate_raw_config(make_raw_config(**overrides))


@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0])
def test_overlap_accepts_closed_interval_and_normalizes_to_float(value: object) -> None:
    validated = validate_raw_config(make_raw_config(separation_overlap=value))

    assert validated.separation_overlap == float(value)
    assert isinstance(validated.separation_overlap, float)


@pytest.mark.parametrize("bitrate", ["anything", "0k", "999k"])
def test_non_mp3_bitrate_text_remains_ignored(bitrate: str) -> None:
    validated = validate_raw_config(make_raw_config(output_format="flac", output_bitrate=bitrate))

    assert validated.output_bitrate == bitrate


def test_runtime_resolution_is_frozen_and_preserves_fallback_context() -> None:
    resolution = RuntimeResolution(
        requested="mps",
        actual="cpu",
        backend="ctranslate2",
        fallback_reason="ctranslate2 不支持 MPS，已回退到 CPU",
    )

    assert resolution.used_fallback is True
    with pytest.raises(FrozenInstanceError):
        resolution.actual = "cuda"  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        {"requested": "cpu", "actual": "cuda", "backend": "torch"},
        {"requested": "cuda", "actual": "cpu", "backend": "torch"},
        {"requested": "mps", "actual": "cpu", "backend": "ctranslate2"},
        {
            "requested": "cpu",
            "actual": "cpu",
            "backend": "torch",
            "fallback_reason": "unexpected fallback",
        },
        {
            "requested": "mps",
            "actual": "cpu",
            "backend": "ctranslate2",
            "fallback_reason": "",
        },
    ],
)
def test_runtime_resolution_rejects_contradictory_states(
    values: dict[str, object],
) -> None:
    with pytest.raises(RuntimeResolutionError):
        RuntimeResolution(**values)  # type: ignore[arg-type]


def test_resolved_config_rejects_cross_object_mismatches() -> None:
    disabled = validate_raw_config(make_raw_config())
    enabled = validate_raw_config(make_raw_config(lyrics_enabled=True))
    valid_separation = RuntimeResolution(
        requested="auto",
        actual="cpu",
        backend="torch",
    )
    valid_lyrics = RuntimeResolution(
        requested="auto",
        actual="cpu",
        backend="ctranslate2",
    )

    invalid_cases = [
        {
            "requested": disabled,
            "separation": RuntimeResolution(
                requested="auto",
                actual="cpu",
                backend="ctranslate2",
            ),
            "lyrics": None,
        },
        {
            "requested": disabled,
            "separation": RuntimeResolution(
                requested="cpu",
                actual="cpu",
                backend="torch",
            ),
            "lyrics": None,
        },
        {
            "requested": enabled,
            "separation": valid_separation,
            "lyrics": None,
        },
        {
            "requested": enabled,
            "separation": valid_separation,
            "lyrics": RuntimeResolution(
                requested="auto",
                actual="cpu",
                backend="torch",
            ),
        },
        {
            "requested": enabled,
            "separation": valid_separation,
            "lyrics": RuntimeResolution(
                requested="cpu",
                actual="cpu",
                backend="ctranslate2",
            ),
        },
        {
            "requested": disabled,
            "separation": valid_separation,
            "lyrics": valid_lyrics,
        },
    ]

    for values in invalid_cases:
        with pytest.raises(RuntimeResolutionError):
            ResolvedConfig(**values)  # type: ignore[arg-type]
