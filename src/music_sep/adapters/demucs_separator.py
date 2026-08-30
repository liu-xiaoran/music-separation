from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from music_sep.domain.artifacts import SeparationOptions
from music_sep.domain.errors import DemucsModelError, SeparationAdapterError
from music_sep.ports.separator import SeparatedAudio

logger = logging.getLogger("music_sep")

ModelLoader = Callable[[str], Any]
TrackLoader = Callable[[str, int, int], Any]
ModelApplier = Callable[..., Any]


class DemucsSeparator:
    """使用 Demucs 4.0.1 低级 API 的音源分离适配器。"""

    def __init__(
        self,
        options: SeparationOptions,
        *,
        model_loader: ModelLoader | None = None,
        track_loader: TrackLoader | None = None,
        model_applier: ModelApplier | None = None,
    ) -> None:
        self._options = options
        self._model_loader = model_loader
        self._track_loader = track_loader
        self._model_applier = model_applier
        self._model: Any | None = None

    def _default_model_loader(self, name: str) -> Any:
        try:
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise DemucsModelError(f"无法导入 demucs: {exc}") from exc
        return get_model(name)

    def _default_track_loader(
        self,
        path: str,
        audio_channels: int,
        samplerate: int,
    ) -> Any:
        try:
            from demucs.separate import load_track
        except ImportError as exc:
            raise SeparationAdapterError(f"无法导入 demucs 音频读取器: {exc}") from exc
        return load_track(path, audio_channels, samplerate)

    def _default_model_applier(self, model: object, mix: object, **kwargs: object) -> Any:
        try:
            from demucs.apply import apply_model
        except ImportError as exc:
            raise SeparationAdapterError(f"无法导入 demucs 推理器: {exc}") from exc
        untyped_apply: Any = apply_model
        return untyped_apply(model, mix, **kwargs)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        loader = self._model_loader or self._default_model_loader
        try:
            logger.info(
                "加载 Demucs 模型: %s (设备: %s)",
                self._options.model,
                self._options.device,
            )
            model = loader(self._options.model)
        except DemucsModelError:
            raise
        except Exception as exc:
            raise DemucsModelError(f"模型加载失败: {exc}") from exc

        try:
            model.cpu()
            model.eval()
            self._read_model_metadata(model)
        except Exception as exc:
            raise DemucsModelError(f"模型初始化失败: {exc}") from exc

        self._model = model
        return model

    @staticmethod
    def _read_model_metadata(model: Any) -> tuple[int, int, tuple[str, ...]]:
        try:
            samplerate = int(model.samplerate)
            audio_channels = int(model.audio_channels)
            sources = tuple(str(source) for source in model.sources)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DemucsModelError(f"模型元信息无效: {exc}") from exc

        if samplerate <= 0 or audio_channels <= 0 or not sources:
            raise DemucsModelError("模型元信息无效: 采样率、声道和音轨列表必须非空")
        return samplerate, audio_channels, sources

    def _validate_stem_selection(self, sources: tuple[str, ...]) -> None:
        selected = self._options.stems
        target = self._options.two_stems

        if target is not None and selected is not None:
            raise SeparationAdapterError("--two-stems 和 --stems 不可同时使用")
        if target is not None and target not in sources:
            raise SeparationAdapterError(f"未知音轨: {target}。可用音轨: {', '.join(sources)}")
        if selected is not None:
            if not selected:
                raise SeparationAdapterError("--stems 至少选择一个音轨")
            unknown = [name for name in selected if name not in sources]
            if unknown:
                raise SeparationAdapterError(
                    f"未知音轨: {', '.join(unknown)}。可用音轨: {', '.join(sources)}"
                )

    @staticmethod
    def _normalize_audio(wav: Any) -> tuple[Any, Any, Any]:
        import torch

        if not isinstance(wav, torch.Tensor):
            raise SeparationAdapterError("Demucs 音频读取器未返回 Tensor")
        if wav.ndim != 2 or wav.shape[0] == 0 or wav.shape[1] == 0:
            raise SeparationAdapterError(
                f"输入音频 Tensor 必须为非空二维 (channels, samples)，当前形状: {tuple(wav.shape)}"
            )
        if not wav.dtype.is_floating_point:
            wav = wav.float()
        if not torch.isfinite(wav).all():
            raise SeparationAdapterError("输入音频包含 NaN 或 Inf")

        reference = wav.mean(dim=0)
        offset = reference.mean()
        scale = reference.std()
        if not torch.isfinite(scale) or scale.item() == 0:
            scale = torch.ones_like(scale)

        normalized = (wav - offset) / scale
        if not torch.isfinite(normalized).all():
            raise SeparationAdapterError("音频归一化产生 NaN 或 Inf")
        return normalized, offset, scale

    @staticmethod
    def _validate_estimates(estimated: Any, source_count: int, wav: Any) -> Any:
        import torch

        if not isinstance(estimated, torch.Tensor):
            raise SeparationAdapterError("Demucs 推理器未返回 Tensor")
        if estimated.ndim != 4 or estimated.shape[0] != 1:
            raise SeparationAdapterError(
                "Demucs 输出形状无效: 期望 (1, sources, channels, samples)，"
                f"实际 {tuple(estimated.shape)}"
            )
        estimated = estimated[0]
        if estimated.shape[0] != source_count:
            raise SeparationAdapterError(
                f"Demucs 输出音轨数量无效: 期望 {source_count}，实际 {estimated.shape[0]}"
            )
        if tuple(estimated.shape[1:]) != tuple(wav.shape):
            raise SeparationAdapterError(
                "Demucs 输出声道或采样长度无效: "
                f"期望 {tuple(wav.shape)}，实际 {tuple(estimated.shape[1:])}"
            )
        return estimated

    def separate(self, input_file: Path) -> SeparatedAudio:
        import torch

        model = self._load_model()
        samplerate, audio_channels, sources = self._read_model_metadata(model)
        self._validate_stem_selection(sources)

        track_loader = self._track_loader or self._default_track_loader
        model_applier = self._model_applier or self._default_model_applier

        try:
            logger.info("开始分离: %s", input_file.name)
            wav = track_loader(str(input_file), audio_channels, samplerate)
            normalized, offset, scale = self._normalize_audio(wav)
            estimated = model_applier(
                model,
                normalized.unsqueeze(0),
                shifts=self._options.shifts,
                overlap=self._options.overlap,
                progress=True,
                device=self._options.device,
            )
            estimated = self._validate_estimates(estimated, len(sources), normalized)
            estimated = estimated * scale.to(estimated.device, estimated.dtype)
            estimated = estimated + offset.to(estimated.device, estimated.dtype)
            if not torch.isfinite(estimated).all():
                raise SeparationAdapterError("Demucs 反归一化结果包含 NaN 或 Inf")
        except torch.cuda.OutOfMemoryError as exc:
            raise SeparationAdapterError(
                "GPU 显存不足。尝试使用 --device cpu 或减小 --shifts。"
            ) from exc
        except SystemExit as exc:
            raise SeparationAdapterError(f"无法读取音频文件: {input_file}") from exc
        except SeparationAdapterError:
            raise
        except Exception as exc:
            raise SeparationAdapterError(f"分离失败: {exc}") from exc

        separated = {name: estimated[index] for index, name in enumerate(sources)}
        if self._options.two_stems is not None:
            target = self._options.two_stems
            target_tensor = separated[target]
            no_target = torch.zeros_like(target_tensor)
            for name in sources:
                if name != target:
                    no_target += separated[name]
            separated = {target: target_tensor, f"no_{target}": no_target}
        elif self._options.stems is not None:
            selected = set(self._options.stems)
            separated = {name: separated[name] for name in sources if name in selected}

        return SeparatedAudio(samplerate=samplerate, sources=separated)

    def get_model_info(self) -> dict[str, object]:
        model = self._load_model()
        samplerate, _, sources = self._read_model_metadata(model)
        return {
            "model": self._options.model,
            "samplerate": samplerate,
            "sources": list(sources),
        }
