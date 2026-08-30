class AdapterError(Exception):
    """第三方适配层错误。"""


class DemucsModelError(AdapterError):
    """Demucs 模型导入、加载或初始化失败。"""


class SeparationAdapterError(AdapterError):
    """Demucs 音频读取、推理或结果校验失败。"""


class AudioWriteError(AdapterError):
    """音频输出参数或编码失败。"""
