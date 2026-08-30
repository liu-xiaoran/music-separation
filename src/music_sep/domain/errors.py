class DomainError(Exception):
    """应用核心可识别的内部错误。"""


class ConfigValidationError(DomainError):
    """合并后的逻辑配置不满足领域约束。"""


class RuntimeResolutionError(DomainError):
    """请求的运行设备或后端无法解析。"""


class AdapterError(DomainError):
    """第三方适配层错误。"""


class DemucsModelError(AdapterError):
    """Demucs 模型导入、加载或初始化失败。"""


class SeparationAdapterError(AdapterError):
    """Demucs 音频读取、推理或结果校验失败。"""


class AudioWriteError(AdapterError):
    """音频输出参数或编码失败。"""
