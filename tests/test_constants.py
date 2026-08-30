from music_sep.constants import (
    AVAILABLE_MODELS,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_INPUT_EXTENSIONS,
    VIZ_TYPES,
    WHISPER_MODELS,
)
from music_sep.domain.catalog import (
    DEMUCS_MODEL_DESCRIPTIONS,
    SUPPORTED_INPUT_EXTENSIONS as DOMAIN_INPUT_EXTENSIONS,
    VISUALIZATION_TYPES,
    WHISPER_MODELS as DOMAIN_WHISPER_MODELS,
)


def test_legacy_constants_preserve_values_and_container_types() -> None:
    assert type(AVAILABLE_MODELS) is dict
    assert AVAILABLE_MODELS == dict(DEMUCS_MODEL_DESCRIPTIONS)
    assert type(WHISPER_MODELS) is tuple
    assert WHISPER_MODELS == DOMAIN_WHISPER_MODELS
    assert type(VIZ_TYPES) is tuple
    assert VIZ_TYPES == VISUALIZATION_TYPES
    assert type(SUPPORTED_INPUT_EXTENSIONS) is frozenset
    assert SUPPORTED_INPUT_EXTENSIONS == DOMAIN_INPUT_EXTENSIONS
    assert DEFAULT_SAMPLE_RATE == 44100
