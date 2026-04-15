import pytest
from pathlib import Path
import tempfile
import shutil
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@pytest.fixture
def tmp_dir(tmp_path):
    """临时目录"""
    return tmp_path


@pytest.fixture
def sample_audio(tmp_path):
    """生成一个短的测试 WAV 音频文件"""
    sr = 44100
    duration = 1.0  # 1 秒
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 生成 440Hz 正弦波，双声道
    y = np.column_stack([
        np.sin(2 * np.pi * 440 * t),
        np.sin(2 * np.pi * 440 * t),
    ])
    audio_path = tmp_path / "test_audio.wav"
    sf.write(str(audio_path), y, sr)
    return audio_path


@pytest.fixture
def sample_mp3_path(tmp_path):
    """返回一个假的 mp3 路径（不创建实际文件）"""
    return tmp_path / "test.mp3"


@pytest.fixture
def toml_config_file(tmp_path):
    """创建一个测试用的 TOML 配置文件"""
    config_content = """
[separation]
model = "htdemucs_ft"
device = "cpu"
shifts = 2
overlap = 0.5

[lyrics]
enabled = true
whisper_model = "small"
whisper_device = "cpu"
language = "zh"
output_format = "vtt"

[visualization]
enabled = true
types = ["waveform", "mel"]

[output]
output_dir = "output"
format = "flac"
overwrite = true
"""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(config_content)
    return config_path
