from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from music_sep.config import AppConfig
from music_sep.exceptions import AudioProcessingError, MusicSepError
from music_sep.lyrics import LyricsTranscriber, TranscriptionResult
from music_sep.outputs import OutputPaths, resolve_output_paths, ensure_output_dirs, OutputExistsError
from music_sep.separation import SeparationEngine
from music_sep.visualization import Visualizer
from music_sep.utils import validate_input_file, detect_device, setup_logging

logger = logging.getLogger("music_sep")


@dataclass
class PipelineResult:
    """流水线执行结果"""
    input_file: Path
    output_paths: OutputPaths
    stems: dict[str, Path] = field(default_factory=dict)
    lyrics_path: Optional[Path] = None
    visualization_paths: Optional[list[Path]] = None


class Pipeline:
    """编排处理阶段"""

    def __init__(self, config: AppConfig):
        self._config = config
        self._separation_engine: Optional[SeparationEngine] = None
        self._transcriber = None  # Step 5 实现
        self._visualizer = None   # Step 6 实现

    def run(self, input_file: Path, dry_run: bool = False) -> PipelineResult:
        """执行完整流水线

        Args:
            input_file: 输入音频文件路径
            dry_run: 仅打印计划，不实际执行

        Returns:
            PipelineResult

        Raises:
            AudioProcessingError: 分离失败
        """
        # === Stage 1: 校验与准备 ===
        logger.info("=" * 50)
        logger.info("Stage 1: 校验与准备")
        logger.info("=" * 50)

        validated_path = validate_input_file(input_file)
        logger.info(f"输入文件: {validated_path}")

        # 解析设备
        self._config.separation.device = detect_device(
            self._config.separation.device, backend="torch"
        )
        logger.info(f"分离设备: {self._config.separation.device}")

        # 解析输出路径
        try:
            output_paths = resolve_output_paths(
                validated_path,
                self._config.output.output_dir,
                lyrics_format=self._config.lyrics.output_format,
                overwrite=self._config.output.overwrite,
            )
        except OutputExistsError as e:
            logger.warning(str(e))
            raise

        if dry_run:
            self._print_dry_run_plan(validated_path, output_paths)
            return PipelineResult(
                input_file=validated_path,
                output_paths=output_paths,
            )

        # 创建输出目录
        ensure_output_dirs(output_paths, overwrite=self._config.output.overwrite)
        logger.info(f"输出目录: {output_paths.base_dir}")

        # === Stage 2: 音乐分离 ===
        logger.info("=" * 50)
        logger.info("Stage 2: 音乐分离")
        logger.info("=" * 50)

        result = PipelineResult(
            input_file=validated_path,
            output_paths=output_paths,
        )

        self._separation_engine = SeparationEngine(self._config.separation)
        stems = self._separation_engine.separate(
            validated_path,
            output_paths,
            output_format=self._config.output.format,
            bitrate=self._config.output.bitrate,
        )
        result.stems = stems

        # === Stage 3: 歌词识别（可选） ===
        if self._config.lyrics.enabled:
            logger.info("=" * 50)
            logger.info("Stage 3: 歌词识别")
            logger.info("=" * 50)
            try:
                # 歌词设备独立于分离设备
                self._config.lyrics.whisper_device = detect_device(
                    self._config.lyrics.whisper_device, backend="ctranslate2"
                )
                transcriber = LyricsTranscriber(self._config.lyrics)
                # 优先使用分离出的人声轨
                audio_source = result.stems.get("vocals", validated_path)
                if "vocals" in result.stems:
                    logger.info("使用分离出的人声轨进行歌词识别")
                else:
                    logger.info("未找到人声轨，使用原始音频进行歌词识别")

                transcription = transcriber.transcribe(audio_source)
                lyrics_path = transcriber.save_lyrics(
                    transcription,
                    output_paths.lyrics_path,
                    fmt=self._config.lyrics.output_format,
                )
                result.lyrics_path = lyrics_path
            except Exception as e:
                logger.error(f"歌词识别失败: {e}")
                logger.info("跳过歌词识别，继续后续处理")

        # === Stage 4: 可视化（可选） ===
        if self._config.visualization.enabled:
            logger.info("=" * 50)
            logger.info("Stage 4: 可视化")
            logger.info("=" * 50)
            try:
                visualizer = Visualizer(self._config.visualization)
                viz_paths: list[Path] = []

                # 对原始文件生成可视化
                original_viz = visualizer.generate(
                    validated_path, output_paths.viz_dir, stem_name=None
                )
                viz_paths.extend(original_viz)

                # 对每个 stem 生成可视化
                for stem_name, stem_path in result.stems.items():
                    stem_viz = visualizer.generate(
                        stem_path, output_paths.viz_dir, stem_name=stem_name
                    )
                    viz_paths.extend(stem_viz)

                result.visualization_paths = viz_paths
            except Exception as e:
                logger.error(f"可视化生成失败: {e}")
                logger.info("跳过可视化")

        logger.info("=" * 50)
        logger.info("处理完成!")
        logger.info("=" * 50)
        self._print_summary(result)

        return result

    def _print_dry_run_plan(self, input_file: Path, output_paths: OutputPaths) -> None:
        """打印 dry-run 执行计划"""
        sep = self._config.separation
        out = self._config.output
        lyr = self._config.lyrics
        viz = self._config.visualization

        print("\n📋 执行计划 (dry-run)")
        print("=" * 50)
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_paths.base_dir}")
        print()
        print(f"[分离]")
        print(f"  模型: {sep.model}")
        print(f"  设备: {sep.device}")
        print(f"  偏移次数: {sep.shifts}")
        print(f"  重叠比例: {sep.overlap}")
        if sep.two_stems:
            print(f"  双轨模式: {sep.two_stems}")
        if sep.stems:
            print(f"  输出音轨: {', '.join(sep.stems)}")
        print(f"  输出格式: {out.format}")
        if out.format == "mp3":
            print(f"  比特率: {out.bitrate}")

        if lyr.enabled:
            print()
            print(f"[歌词识别]")
            print(f"  Whisper 模型: {lyr.whisper_model}")
            print(f"  设备: {lyr.whisper_device}")
            if lyr.language:
                print(f"  语言: {lyr.language}")
            print(f"  输出格式: {lyr.output_format}")

        if viz.enabled:
            print()
            print(f"[可视化]")
            print(f"  类型: {', '.join(viz.types)}")

        print("=" * 50)
        print("（dry-run 模式，未执行任何操作）\n")

    def _print_summary(self, result: PipelineResult) -> None:
        """打印处理结果摘要"""
        print(f"\n输出目录: {result.output_paths.base_dir}")
        print(f"分离音轨: {', '.join(result.stems.keys())}")
        if result.lyrics_path:
            print(f"歌词文件: {result.lyrics_path.name}")
        if result.visualization_paths:
            print(f"可视化图片: {len(result.visualization_paths)} 张")
