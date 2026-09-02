from typing import Protocol

from music_sep.domain.config import (
    DevicePreference,
    RuntimeBackend,
    RuntimeResolution,
)


class RuntimeResolver(Protocol):
    """按第三方后端能力解析一次运行所使用的设备。"""

    def resolve(
        self,
        preference: DevicePreference,
        *,
        backend: RuntimeBackend,
    ) -> RuntimeResolution: ...
