from types import MappingProxyType

DEMUCS_MODEL_DESCRIPTIONS = MappingProxyType(
    {
        "htdemucs": "Hybrid Transformer 4轨 (drums, bass, other, vocals)",
        "htdemucs_6s": "Hybrid Transformer 6轨 (+ guitar, piano)",
        "htdemucs_ft": "微调版 Hybrid Transformer 4轨",
        "mdx": "MDX 挑战基线 4轨",
        "mdx_extra": "MDX 增强 4轨",
        "hdemucs_mmi": "Hybrid Demucs 2轨 (vocals + accompaniment)",
    }
)
WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "distil-large-v3",
    "turbo",
)
VISUALIZATION_TYPES = ("waveform", "spectrogram", "mel")
LYRICS_FORMATS = ("srt", "lrc", "vtt", "txt", "json")
OUTPUT_FORMATS = ("wav", "mp3", "flac")
DEVICE_PREFERENCES = ("auto", "cpu", "cuda", "mps")
RUNTIME_BACKENDS = ("torch", "ctranslate2")
SUPPORTED_INPUT_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"})
