from pathlib import Path
from typing import Any

import pytest
import torch

from music_sep.adapters.demucs_separator import DemucsSeparator
from music_sep.domain.artifacts import SeparationOptions
from music_sep.domain.errors import DemucsModelError, SeparationAdapterError


class FakeModel:
    def __init__(self, sources: tuple[str, ...] = ("drums", "bass", "other", "vocals")) -> None:
        self.sources = list(sources)
        self.samplerate = 44100
        self.audio_channels = 2
        self.cpu_calls = 0
        self.eval_calls = 0
        self.cpu_error: Exception | None = None

    def cpu(self) -> "FakeModel":
        self.cpu_calls += 1
        if self.cpu_error is not None:
            raise self.cpu_error
        return self

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self


class DemucsHarness:
    def __init__(
        self,
        wav: torch.Tensor,
        normalized_sources: torch.Tensor,
        model: FakeModel | None = None,
    ) -> None:
        self.wav = wav
        self.normalized_sources = normalized_sources
        self.model = model or FakeModel()
        self.model_loader_calls: list[str] = []
        self.track_loader_calls: list[tuple[str, int, int]] = []
        self.apply_calls: list[tuple[object, torch.Tensor, dict[str, Any]]] = []
        self.model_error: Exception | None = None
        self.track_error: BaseException | None = None
        self.apply_error: Exception | None = None

    def model_loader(self, name: str) -> FakeModel:
        self.model_loader_calls.append(name)
        if self.model_error is not None:
            raise self.model_error
        return self.model

    def track_loader(self, path: str, audio_channels: int, samplerate: int) -> torch.Tensor:
        self.track_loader_calls.append((path, audio_channels, samplerate))
        if self.track_error is not None:
            raise self.track_error
        return self.wav.clone()

    def model_applier(
        self,
        model: object,
        mix: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.apply_calls.append((model, mix.detach().clone(), kwargs))
        if self.apply_error is not None:
            raise self.apply_error
        return self.normalized_sources.unsqueeze(0).clone()


def options(**overrides: object) -> SeparationOptions:
    values: dict[str, object] = {
        "model": "htdemucs",
        "device": "cpu",
        "shifts": 2,
        "overlap": 0.4,
        "two_stems": None,
        "stems": None,
    }
    values.update(overrides)
    return SeparationOptions(**values)  # type: ignore[arg-type]


def make_separator(harness: DemucsHarness, **overrides: object) -> DemucsSeparator:
    return DemucsSeparator(
        options(**overrides),
        model_loader=harness.model_loader,
        track_loader=harness.track_loader,
        model_applier=harness.model_applier,
    )


def official_denormalize(wav: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    reference = wav.mean(dim=0)
    return sources * reference.std() + reference.mean()


def test_matches_demucs_4_0_1_normalization_and_apply_arguments(tmp_path: Path) -> None:
    wav = torch.tensor(
        [[0.2, 0.4, 0.1, 0.7], [0.4, 0.8, 0.3, 0.5]],
        dtype=torch.float64,
    )
    normalized_sources = torch.arange(32, dtype=torch.float64).reshape(4, 2, 4) / 10
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness, device="mps")
    input_file = tmp_path / "song.wav"

    result = separator.separate(input_file)

    reference = wav.mean(dim=0)
    expected_mix = ((wav - reference.mean()) / reference.std()).unsqueeze(0)
    expected_sources = official_denormalize(wav, normalized_sources)

    assert harness.model_loader_calls == ["htdemucs"]
    assert harness.track_loader_calls == [(str(input_file), 2, 44100)]
    assert len(harness.apply_calls) == 1
    applied_model, applied_mix, kwargs = harness.apply_calls[0]
    assert applied_model is harness.model
    torch.testing.assert_close(applied_mix, expected_mix)
    assert kwargs == {
        "shifts": 2,
        "overlap": 0.4,
        "progress": True,
        "device": "mps",
    }
    assert list(result.sources) == harness.model.sources
    for index, name in enumerate(harness.model.sources):
        assert isinstance(result.sources[name], torch.Tensor)
        torch.testing.assert_close(result.sources[name], expected_sources[index])


def test_model_is_loaded_once_and_prepared_like_official_cli(tmp_path: Path) -> None:
    wav = torch.randn(2, 16)
    sources = torch.randn(4, 2, 16)
    harness = DemucsHarness(wav, sources)
    separator = make_separator(harness)

    first_info = separator.get_model_info()
    second_info = separator.get_model_info()
    separator.separate(tmp_path / "song.wav")

    assert harness.model_loader_calls == ["htdemucs"]
    assert harness.model.cpu_calls == 1
    assert harness.model.eval_calls == 1
    assert (
        first_info
        == second_info
        == {
            "model": "htdemucs",
            "samplerate": 44100,
            "sources": ["drums", "bass", "other", "vocals"],
        }
    )


def test_constant_input_uses_finite_safe_scale(tmp_path: Path) -> None:
    wav = torch.full((2, 32), 0.25)
    normalized_sources = torch.zeros(4, 2, 32)
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness)

    result = separator.separate(tmp_path / "constant.wav")

    applied_mix = harness.apply_calls[0][1]
    assert torch.count_nonzero(applied_mix) == 0
    assert torch.isfinite(applied_mix).all()
    for source in result.sources.values():
        assert isinstance(source, torch.Tensor)
        assert torch.isfinite(source).all()
        torch.testing.assert_close(source, torch.full((2, 32), 0.25))


def test_low_variance_input_still_matches_official_formula(tmp_path: Path) -> None:
    wav = torch.tensor(
        [
            [0.25, 0.25000001, 0.24999999, 0.25000002],
            [0.25, 0.24999998, 0.25000002, 0.25000001],
        ],
        dtype=torch.float64,
    )
    normalized_sources = torch.ones(4, 2, 4, dtype=torch.float64)
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness)

    result = separator.separate(tmp_path / "quiet.wav")

    expected = official_denormalize(wav, normalized_sources)
    for index, name in enumerate(harness.model.sources):
        source = result.sources[name]
        assert isinstance(source, torch.Tensor)
        torch.testing.assert_close(source, expected[index])
        assert torch.isfinite(source).all()


def test_two_stems_returns_target_and_sum_of_other_denormalized_sources(
    tmp_path: Path,
) -> None:
    wav = torch.tensor([[0.0, 1.0, -1.0, 0.0], [0.0, 1.0, -1.0, 0.0]])
    normalized_sources = torch.stack([torch.full((2, 4), float(value)) for value in (1, 2, 3, 4)])
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness, two_stems="vocals")

    result = separator.separate(tmp_path / "song.wav")

    denormalized = official_denormalize(wav, normalized_sources)
    assert list(result.sources) == ["vocals", "no_vocals"]
    vocals = result.sources["vocals"]
    no_vocals = result.sources["no_vocals"]
    assert isinstance(vocals, torch.Tensor)
    assert isinstance(no_vocals, torch.Tensor)
    torch.testing.assert_close(vocals, denormalized[3])
    torch.testing.assert_close(no_vocals, denormalized[:3].sum(dim=0))


def test_stems_filter_preserves_model_source_order(tmp_path: Path) -> None:
    wav = torch.randn(2, 16)
    normalized_sources = torch.randn(4, 2, 16)
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness, stems=("vocals", "drums"))

    result = separator.separate(tmp_path / "song.wav")

    assert list(result.sources) == ["drums", "vocals"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stems": ()}, "至少选择一个音轨"),
        ({"stems": ("vocals", "missing")}, "未知音轨"),
        ({"two_stems": "missing"}, "未知音轨"),
        (
            {"two_stems": "vocals", "stems": ("vocals",)},
            "不可同时使用",
        ),
    ],
)
def test_invalid_stem_selection_fails_before_inference(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    wav = torch.randn(2, 16)
    normalized_sources = torch.randn(4, 2, 16)
    harness = DemucsHarness(wav, normalized_sources)
    separator = make_separator(harness, **overrides)

    with pytest.raises(SeparationAdapterError, match=message):
        separator.separate(tmp_path / "song.wav")

    assert harness.apply_calls == []


def test_model_load_error_is_wrapped_with_cause() -> None:
    harness = DemucsHarness(torch.randn(2, 8), torch.randn(4, 2, 8))
    harness.model_error = RuntimeError("network down")
    separator = make_separator(harness)

    with pytest.raises(DemucsModelError, match="模型加载失败") as exc_info:
        separator.get_model_info()

    assert exc_info.value.__cause__ is harness.model_error


def test_model_preparation_error_is_wrapped_with_cause() -> None:
    harness = DemucsHarness(torch.randn(2, 8), torch.randn(4, 2, 8))
    harness.model.cpu_error = RuntimeError("device failure")
    separator = make_separator(harness)

    with pytest.raises(DemucsModelError, match="模型初始化失败") as exc_info:
        separator.get_model_info()

    assert exc_info.value.__cause__ is harness.model.cpu_error


def test_load_track_system_exit_is_converted_to_processing_error(tmp_path: Path) -> None:
    harness = DemucsHarness(torch.randn(2, 8), torch.randn(4, 2, 8))
    harness.track_error = SystemExit(1)
    separator = make_separator(harness)

    with pytest.raises(SeparationAdapterError, match="无法读取音频") as exc_info:
        separator.separate(tmp_path / "bad.wav")

    assert exc_info.value.__cause__ is harness.track_error


def test_cuda_oom_has_actionable_project_message(tmp_path: Path) -> None:
    harness = DemucsHarness(torch.randn(2, 8), torch.randn(4, 2, 8))
    harness.apply_error = torch.cuda.OutOfMemoryError("oom")
    separator = make_separator(harness, device="cuda")

    with pytest.raises(SeparationAdapterError, match="GPU 显存不足") as exc_info:
        separator.separate(tmp_path / "song.wav")

    assert exc_info.value.__cause__ is harness.apply_error


def test_invalid_model_output_shape_is_rejected(tmp_path: Path) -> None:
    harness = DemucsHarness(torch.randn(2, 8), torch.randn(3, 2, 8))
    separator = make_separator(harness)

    with pytest.raises(SeparationAdapterError, match="音轨数量"):
        separator.separate(tmp_path / "song.wav")
