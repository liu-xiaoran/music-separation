from types import SimpleNamespace
from unittest.mock import patch

import pytest

from music_sep.adapters.torch_runtime import TorchRuntimeResolver
from music_sep.domain.config import RuntimeResolution
from music_sep.domain.errors import RuntimeResolutionError
from music_sep.exceptions import DeviceNotAvailableError
from music_sep.utils import detect_device, resolve_device


def make_resolver(
    *,
    cuda: bool,
    mps: bool,
    ctranslate2_cuda: bool | None = None,
) -> TorchRuntimeResolver:
    return TorchRuntimeResolver(
        cuda_available=lambda: cuda,
        mps_available=lambda: mps,
        ctranslate2_cuda_available=lambda: cuda if ctranslate2_cuda is None else ctranslate2_cuda,
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_torch_auto_uses_cuda_then_mps_then_cpu(
    cuda: bool,
    mps: bool,
    expected: str,
) -> None:
    resolution = make_resolver(cuda=cuda, mps=mps).resolve("auto", backend="torch")

    assert resolution == RuntimeResolution(
        requested="auto",
        actual=expected,
        backend="torch",
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected", "has_reason"),
    [
        (True, True, "cuda", False),
        (True, False, "cuda", False),
        (False, True, "cpu", True),
        (False, False, "cpu", False),
    ],
)
def test_ctranslate2_auto_never_selects_mps(
    cuda: bool,
    mps: bool,
    expected: str,
    has_reason: bool,
) -> None:
    resolution = make_resolver(cuda=cuda, mps=mps).resolve(
        "auto",
        backend="ctranslate2",
    )

    assert resolution.requested == "auto"
    assert resolution.actual == expected
    assert resolution.backend == "ctranslate2"
    assert (resolution.fallback_reason is not None) is has_reason
    if has_reason:
        assert "MPS" in resolution.fallback_reason
        assert "CPU" in resolution.fallback_reason


@pytest.mark.parametrize(
    ("preference", "backend", "cuda", "mps", "ctranslate2_cuda"),
    [
        ("cuda", "torch", True, False, False),
        ("cuda", "ctranslate2", False, False, True),
        ("mps", "torch", False, True, False),
        ("cpu", "ctranslate2", False, False, False),
    ],
)
def test_explicit_available_devices_are_preserved(
    preference: str,
    backend: str,
    cuda: bool,
    mps: bool,
    ctranslate2_cuda: bool,
) -> None:
    resolution = make_resolver(
        cuda=cuda,
        mps=mps,
        ctranslate2_cuda=ctranslate2_cuda,
    ).resolve(preference, backend=backend)

    assert resolution.requested == preference
    assert resolution.actual == preference
    assert resolution.backend == backend
    assert resolution.fallback_reason is None


def test_ctranslate2_cuda_capability_is_independent_from_pytorch() -> None:
    resolver = make_resolver(cuda=True, mps=False, ctranslate2_cuda=False)

    assert resolver.resolve("auto", backend="torch").actual == "cuda"
    assert resolver.resolve("auto", backend="ctranslate2").actual == "cpu"
    with pytest.raises(RuntimeResolutionError, match="CUDA"):
        resolver.resolve("cuda", backend="ctranslate2")


def test_default_ctranslate2_probe_uses_backend_device_count() -> None:
    resolver = TorchRuntimeResolver(
        cuda_available=lambda: False,
        mps_available=lambda: False,
    )
    module = SimpleNamespace(get_cuda_device_count=lambda: 1)

    with patch(
        "music_sep.adapters.torch_runtime.import_module",
        return_value=module,
    ) as import_module:
        resolution = resolver.resolve("auto", backend="ctranslate2")

    assert resolution.actual == "cuda"
    import_module.assert_called_once_with("ctranslate2")


def test_explicit_ctranslate2_mps_falls_back_to_cpu_with_reason() -> None:
    resolution = make_resolver(cuda=False, mps=True).resolve(
        "mps",
        backend="ctranslate2",
    )

    assert resolution == RuntimeResolution(
        requested="mps",
        actual="cpu",
        backend="ctranslate2",
        fallback_reason="ctranslate2 不支持 MPS，已回退到 CPU",
    )


@pytest.mark.parametrize(
    ("preference", "backend", "cuda", "mps"),
    [
        ("cuda", "torch", False, False),
        ("mps", "torch", False, False),
        ("cuda", "ctranslate2", False, True),
    ],
)
def test_unavailable_explicit_device_raises_domain_error(
    preference: str,
    backend: str,
    cuda: bool,
    mps: bool,
) -> None:
    with pytest.raises(RuntimeResolutionError, match=preference.upper()):
        make_resolver(cuda=cuda, mps=mps).resolve(preference, backend=backend)


@pytest.mark.parametrize(
    ("preference", "backend"),
    [
        ("tpu", "torch"),
        ("cpu", "unknown"),
        ("auto", "unknown"),
    ],
)
def test_unknown_preference_or_backend_is_always_rejected(
    preference: str,
    backend: str,
) -> None:
    with pytest.raises(RuntimeResolutionError):
        make_resolver(cuda=False, mps=False).resolve(preference, backend=backend)


def test_resolve_device_exposes_structured_result() -> None:
    resolver = make_resolver(cuda=False, mps=True)

    resolution = resolve_device("auto", backend="ctranslate2", resolver=resolver)

    assert resolution.requested == "auto"
    assert resolution.actual == "cpu"
    assert resolution.used_fallback is True


def test_fallback_warning_escapes_terminal_control_characters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ControlSequenceFallbackResolver:
        def resolve(self, preference: str, *, backend: str) -> RuntimeResolution:
            return RuntimeResolution(
                requested="auto",
                actual="cpu",
                backend="ctranslate2",
                fallback_reason=f"{chr(27)}[2J",
            )

    with caplog.at_level("WARNING", logger="music_sep"):
        resolve_device(
            "auto",
            backend="ctranslate2",
            resolver=ControlSequenceFallbackResolver(),
        )

    assert "\x1b" not in caplog.text
    assert "\\x1b[2J" in caplog.text


def test_detect_device_preserves_legacy_string_and_maps_domain_error() -> None:
    resolver = make_resolver(cuda=False, mps=True)

    assert detect_device("mps", backend="ctranslate2", resolver=resolver) == "cpu"
    with pytest.raises(DeviceNotAvailableError) as exc_info:
        detect_device("cuda", backend="torch", resolver=resolver)

    assert isinstance(exc_info.value.__cause__, RuntimeResolutionError)


class FalsyCapabilityCheck:
    def __bool__(self) -> bool:
        return False

    def __call__(self) -> bool:
        return True


class FalsyResolver:
    def __bool__(self) -> bool:
        return False

    def resolve(self, preference: str, *, backend: str) -> RuntimeResolution:
        return RuntimeResolution(
            requested="cpu",
            actual="cpu",
            backend="torch",
        )


def test_falsy_injected_capability_check_is_not_discarded() -> None:
    resolver = TorchRuntimeResolver(
        cuda_available=FalsyCapabilityCheck(),
        mps_available=lambda: False,
        ctranslate2_cuda_available=lambda: False,
    )

    assert resolver.resolve("cuda", backend="torch").actual == "cuda"


def test_falsy_injected_resolver_is_not_discarded() -> None:
    resolver = FalsyResolver()

    resolution = resolve_device("cpu", backend="torch", resolver=resolver)

    assert resolution.actual == "cpu"
