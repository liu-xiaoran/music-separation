from typer.testing import CliRunner

from music_sep.cli import app

runner = CliRunner()


class TestCLIBasic:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Music Separation" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_separate_help(self):
        result = runner.invoke(app, ["separate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--lyrics" in result.output
        assert "--visualize" in result.output

    def test_models_command(self):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "htdemucs" in result.output

    def test_info_nonexistent_file(self):
        result = runner.invoke(app, ["info", "/nonexistent/file.mp3"])
        assert result.exit_code != 0

    def test_separate_nonexistent_file(self):
        result = runner.invoke(app, ["separate", "/nonexistent/file.mp3"])
        assert result.exit_code != 0

    def test_dry_run(self, sample_audio):
        result = runner.invoke(app, ["separate", str(sample_audio), "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.output

    def test_dry_run_with_lyrics_and_visualize(self, sample_audio):
        result = runner.invoke(
            app,
            [
                "separate",
                str(sample_audio),
                "--dry-run",
                "--lyrics",
                "--visualize",
            ],
        )
        assert result.exit_code == 0
        assert "歌词识别" in result.output
        assert "可视化" in result.output

    def test_invalid_model(self, sample_audio):
        result = runner.invoke(
            app,
            [
                "separate",
                str(sample_audio),
                "--model",
                "nonexistent",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0
