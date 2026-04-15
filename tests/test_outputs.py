import pytest
from pathlib import Path

from music_sep.outputs import (
    OutputPaths,
    resolve_output_paths,
    ensure_output_dirs,
    stem_filename,
    viz_filename,
    OutputExistsError,
)


class TestResolveOutputPaths:
    def test_basic_paths(self, tmp_path):
        input_file = tmp_path / "song.mp3"
        input_file.touch()

        paths = resolve_output_paths(input_file, tmp_path / "output")
        assert paths.base_dir == tmp_path / "output" / "song"
        assert paths.stems_dir == tmp_path / "output" / "song" / "stems"
        assert paths.lyrics_path == tmp_path / "output" / "song" / "lyrics.srt"
        assert paths.viz_dir == tmp_path / "output" / "song" / "visualizations"

    def test_custom_lyrics_format(self, tmp_path):
        input_file = tmp_path / "song.mp3"
        input_file.touch()

        paths = resolve_output_paths(input_file, tmp_path / "output", lyrics_format="txt")
        assert paths.lyrics_path == tmp_path / "output" / "song" / "lyrics.txt"

    def test_existing_dir_raises(self, tmp_path):
        input_file = tmp_path / "song.mp3"
        input_file.touch()
        # 创建已存在的输出目录
        (tmp_path / "output" / "song").mkdir(parents=True)

        with pytest.raises(OutputExistsError, match="输出目录已存在"):
            resolve_output_paths(input_file, tmp_path / "output")

    def test_overwrite_existing_dir(self, tmp_path):
        input_file = tmp_path / "song.mp3"
        input_file.touch()
        # 创建已存在的输出目录，带一个文件
        existing = tmp_path / "output" / "song"
        existing.mkdir(parents=True)
        (existing / "old_file.txt").write_text("old")

        paths = resolve_output_paths(
            input_file, tmp_path / "output", overwrite=True
        )
        assert paths.base_dir == existing


class TestEnsureOutputDirs:
    def test_creates_dirs(self, tmp_path):
        paths = OutputPaths(
            base_dir=tmp_path / "output" / "song",
            stems_dir=tmp_path / "output" / "song" / "stems",
            lyrics_path=tmp_path / "output" / "song" / "lyrics.srt",
            viz_dir=tmp_path / "output" / "song" / "visualizations",
        )

        ensure_output_dirs(paths)

        assert paths.stems_dir.exists()
        assert paths.viz_dir.exists()

    def test_overwrite_removes_existing(self, tmp_path):
        base = tmp_path / "output" / "song"
        base.mkdir(parents=True)
        (base / "old_file.txt").write_text("old")

        paths = OutputPaths(
            base_dir=base,
            stems_dir=base / "stems",
            lyrics_path=base / "lyrics.srt",
            viz_dir=base / "visualizations",
        )

        ensure_output_dirs(paths, overwrite=True)

        assert not (base / "old_file.txt").exists()
        assert paths.stems_dir.exists()


class TestFilenameHelpers:
    def test_stem_filename(self):
        assert stem_filename("vocals", "wav") == "vocals.wav"
        assert stem_filename("drums", "mp3") == "drums.mp3"

    def test_viz_filename_with_stem(self):
        assert viz_filename("waveform", "vocals") == "waveform_vocals.png"

    def test_viz_filename_without_stem(self):
        assert viz_filename("spectrogram") == "spectrogram_original.png"
