import pytest
import json

from music_sep.config import LyricsConfig
from music_sep.lyrics import LyricsTranscriber, TranscriptionResult
from music_sep.exceptions import TranscriptionError


@pytest.fixture
def lyrics_config():
    return LyricsConfig(
        enabled=True,
        whisper_model="tiny",
        whisper_device="cpu",
        language=None,
        output_format="srt",
    )


@pytest.fixture
def sample_result():
    return TranscriptionResult(
        segments=[
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 3.0, "end": 5.5, "text": "This is a test"},
        ],
        language="en",
        language_probability=0.95,
    )


class TestTranscriptionResult:
    def test_fields(self, sample_result):
        assert len(sample_result.segments) == 2
        assert sample_result.language == "en"
        assert sample_result.language_probability == 0.95


class TestLyricsTranscriber:
    def test_init(self, lyrics_config):
        transcriber = LyricsTranscriber(lyrics_config)
        assert transcriber._model is None


class TestSaveLyrics:
    def test_srt_format(self, lyrics_config, sample_result, tmp_path):
        transcriber = LyricsTranscriber(lyrics_config)
        output_path = tmp_path / "lyrics.srt"
        result = transcriber.save_lyrics(sample_result, output_path, fmt="srt")

        assert result == output_path
        content = output_path.read_text()
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Hello world" in content
        assert "1\n" in content

    def test_vtt_format(self, lyrics_config, sample_result, tmp_path):
        transcriber = LyricsTranscriber(lyrics_config)
        output_path = tmp_path / "lyrics.vtt"
        transcriber.save_lyrics(sample_result, output_path, fmt="vtt")

        content = output_path.read_text()
        assert content.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in content

    def test_txt_format(self, lyrics_config, sample_result, tmp_path):
        transcriber = LyricsTranscriber(lyrics_config)
        output_path = tmp_path / "lyrics.txt"
        transcriber.save_lyrics(sample_result, output_path, fmt="txt")

        content = output_path.read_text()
        assert "Hello world" in content
        assert "This is a test" in content
        # 纯文本不含时间戳
        assert "-->" not in content

    def test_json_format(self, lyrics_config, sample_result, tmp_path):
        transcriber = LyricsTranscriber(lyrics_config)
        output_path = tmp_path / "lyrics.json"
        transcriber.save_lyrics(sample_result, output_path, fmt="json")

        data = json.loads(output_path.read_text())
        assert len(data) == 2
        assert data[0]["text"] == "Hello world"
        assert data[0]["start"] == 0.0

    def test_unsupported_format(self, lyrics_config, sample_result, tmp_path):
        transcriber = LyricsTranscriber(lyrics_config)
        with pytest.raises(TranscriptionError, match="不支持的歌词格式"):
            transcriber.save_lyrics(sample_result, tmp_path / "lyrics.pdf", fmt="pdf")

    def test_chinese_text(self, lyrics_config, tmp_path):
        result = TranscriptionResult(
            segments=[{"start": 0.0, "end": 2.0, "text": "你好世界"}],
            language="zh",
            language_probability=0.9,
        )
        transcriber = LyricsTranscriber(lyrics_config)
        output_path = tmp_path / "lyrics.srt"
        transcriber.save_lyrics(result, output_path, fmt="srt")

        content = output_path.read_text(encoding="utf-8")
        assert "你好世界" in content
