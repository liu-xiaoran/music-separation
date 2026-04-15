class MusicSepError(Exception):
    """基础异常"""


class UnsupportedFormatError(MusicSepError):
    """不支持的输入文件格式"""


class DeviceNotAvailableError(MusicSepError):
    """请求的计算设备不可用"""


class ModelNotFoundError(MusicSepError):
    """模型不存在或下载失败"""


class AudioProcessingError(MusicSepError):
    """Demucs 分离过程中的错误"""


class TranscriptionError(MusicSepError):
    """Whisper 歌词识别过程中的错误"""


class VisualizationError(MusicSepError):
    """可视化生成过程中的错误"""


class ConfigurationError(MusicSepError):
    """无效的配置值"""
