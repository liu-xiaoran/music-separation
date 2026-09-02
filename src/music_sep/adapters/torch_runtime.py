from __future__ import annotations

from importlib import import_module
from typing import Callable, cast

from music_sep.domain.catalog import DEVICE_PREFERENCES, RUNTIME_BACKENDS
from music_sep.domain.config import DevicePreference, RuntimeBackend, RuntimeResolution
from music_sep.domain.errors import RuntimeResolutionError

CapabilityCheck = Callable[[], bool]


def _torch_module() -> object:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise RuntimeResolutionError(f"无法导入 PyTorch 以检测设备: {exc}") from exc


def _default_cuda_available() -> bool:
    torch = _torch_module()
    try:
        return bool(torch.cuda.is_available())  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeResolutionError(f"检测 CUDA 可用性失败: {exc}") from exc


def _default_mps_available() -> bool:
    torch = _torch_module()
    try:
        backends = torch.backends  # type: ignore[attr-defined]
        return bool(hasattr(backends, "mps") and backends.mps.is_available())
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeResolutionError(f"检测 MPS 可用性失败: {exc}") from exc


def _default_ctranslate2_cuda_available() -> bool:
    try:
        ctranslate2 = import_module("ctranslate2")
    except ImportError:
        return False
    probe = getattr(ctranslate2, "get_cuda_device_count", None)
    if not callable(probe):
        return False
    try:
        return int(probe()) > 0
    except Exception as exc:
        raise RuntimeResolutionError(f"检测 CTranslate2 CUDA 可用性失败: {exc}") from exc


class TorchRuntimeResolver:
    """使用各后端 capability API 解析 Torch/CTranslate2 运行设备。"""

    def __init__(
        self,
        *,
        cuda_available: CapabilityCheck | None = None,
        mps_available: CapabilityCheck | None = None,
        ctranslate2_cuda_available: CapabilityCheck | None = None,
    ) -> None:
        self._cuda_available = (
            cuda_available if cuda_available is not None else _default_cuda_available
        )
        self._mps_available = mps_available if mps_available is not None else _default_mps_available
        self._ctranslate2_cuda_available = (
            ctranslate2_cuda_available
            if ctranslate2_cuda_available is not None
            else _default_ctranslate2_cuda_available
        )

    def resolve(
        self,
        preference: DevicePreference,
        *,
        backend: RuntimeBackend,
    ) -> RuntimeResolution:
        if backend not in RUNTIME_BACKENDS:
            raise RuntimeResolutionError(
                f"未知的后端类型: {backend!r}。支持: {' / '.join(RUNTIME_BACKENDS)}"
            )
        if preference not in DEVICE_PREFERENCES:
            raise RuntimeResolutionError(
                f"未知的设备类型: {preference!r}。支持: {' / '.join(DEVICE_PREFERENCES)}"
            )

        requested = cast(DevicePreference, preference)
        resolved_backend = cast(RuntimeBackend, backend)
        cuda_available = (
            self._cuda_available if backend == "torch" else self._ctranslate2_cuda_available
        )

        if backend == "ctranslate2" and preference == "mps":
            return RuntimeResolution(
                requested=requested,
                actual="cpu",
                backend=resolved_backend,
                fallback_reason="ctranslate2 不支持 MPS，已回退到 CPU",
            )

        if preference == "cpu":
            return RuntimeResolution(
                requested=requested,
                actual="cpu",
                backend=resolved_backend,
            )

        if preference == "cuda":
            if cuda_available():
                return RuntimeResolution(
                    requested=requested,
                    actual="cuda",
                    backend=resolved_backend,
                )
            if backend == "ctranslate2":
                raise RuntimeResolutionError(
                    "CTranslate2 CUDA 不可用。请确认已安装兼容的 NVIDIA 驱动及 CUDA/cuDNN 运行库。"
                )
            raise RuntimeResolutionError(
                "CUDA 不可用。请确认已安装 NVIDIA GPU 和 CUDA 版 PyTorch。"
            )

        if preference == "mps":
            if self._mps_available():
                return RuntimeResolution(
                    requested=requested,
                    actual="mps",
                    backend=resolved_backend,
                )
            raise RuntimeResolutionError(
                "MPS 不可用。请确认使用的是 Apple Silicon Mac 且已安装正确版本的 PyTorch。"
            )

        if cuda_available():
            return RuntimeResolution(
                requested=requested,
                actual="cuda",
                backend=resolved_backend,
            )

        mps_available = self._mps_available()
        if backend == "torch" and mps_available:
            return RuntimeResolution(
                requested=requested,
                actual="mps",
                backend=resolved_backend,
            )
        if backend == "ctranslate2" and mps_available:
            return RuntimeResolution(
                requested=requested,
                actual="cpu",
                backend=resolved_backend,
                fallback_reason="检测到 MPS，但 ctranslate2 不支持 MPS，已回退到 CPU",
            )
        return RuntimeResolution(
            requested=requested,
            actual="cpu",
            backend=resolved_backend,
        )
