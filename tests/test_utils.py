import pytest
import logging
from pathlib import Path

from music_sep.utils import setup_logging, validate_input_file, detect_device, format_duration
from music_sep.exceptions import UnsupportedFormatError, DeviceNotAvailableError


class TestSetupLogging:
    def test_default_level(self):
        logger = setup_logging()
        assert logger.level == logging.INFO

    def test_verbose_level(self):
        logger = setup_logging(verbose=True)
        assert logger.level == logging.DEBUG

    def test_quiet_level(self):
        logger = setup_logging(quiet=True)
        assert logger.level == logging.ERROR

    def test_verbose_overrides_quiet(self):
        logger = setup_logging(verbose=True, quiet=True)
        assert logger.level == logging.DEBUG

    def test_idempotent(self):
        logger1 = setup_logging()
        logger2 = setup_logging()
        assert logger1 is logger2


class TestValidateInputFile:
    def test_valid_wav(self, sample_audio):
        result = validate_input_file(sample_audio)
        assert result.is_absolute()
        assert result.suffix == ".wav"

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_input_file(Path("/nonexistent/audio.mp3"))

    def test_unsupported_format(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        with pytest.raises(UnsupportedFormatError, match="不支持的音频格式"):
            validate_input_file(txt_file)


class TestDetectDevice:
    def test_cpu_always_available(self):
        assert detect_device("cpu") == "cpu"

    def test_auto_returns_valid_device(self):
        device = detect_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_ctranslate2_auto_skips_mps(self):
        device = detect_device("auto", backend="ctranslate2")
        assert device in ("cpu", "cuda")
        # ctranslate2 不应返回 mps
        assert device != "mps"

    def test_explicit_mps_with_ctranslate2_raises(self):
        with pytest.raises(DeviceNotAvailableError, match="ctranslate2"):
            detect_device("mps", backend="ctranslate2")


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0.0) == "00:00"

    def test_normal(self):
        assert format_duration(65.5) == "01:05"

    def test_three_minutes(self):
        assert format_duration(185.0) == "03:05"
