# 支持的输入音频格式
SUPPORTED_INPUT_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
)

# 可用的 Demucs 模型
AVAILABLE_MODELS = {
    "htdemucs": "Hybrid Transformer 4轨 (drums, bass, other, vocals)",
    "htdemucs_6s": "Hybrid Transformer 6轨 (+ guitar, piano)",
    "htdemucs_ft": "微调版 Hybrid Transformer 4轨",
    "mdx": "MDX 挑战基线 4轨",
    "mdx_extra": "MDX 增强 4轨",
    "hdemucs_mmi": "Hybrid Demucs 2轨 (vocals + accompaniment)",
}

# 可用的 Whisper 模型
WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3", "turbo")

# 可用的可视化类型
VIZ_TYPES = ("waveform", "spectrogram", "mel")

# 默认采样率
DEFAULT_SAMPLE_RATE = 44100
