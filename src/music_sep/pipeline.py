from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from music_sep.adapters.torch_runtime import TorchRuntimeResolver
from music_sep.config import AppConfig, to_app_config, to_validated_config
from music_sep.domain.config import ResolvedConfig, RuntimeResolution, ValidatedConfig
from music_sep.lyrics import LyricsTranscriber
from music_sep.outputs import (
    OutputExistsError,
    OutputPaths,
    ensure_output_dirs,
    resolve_output_paths,
)
from music_sep.ports.runtime import RuntimeResolver
from music_sep.separation import SeparationEngine
from music_sep.utils import resolve_device, validate_input_file
from music_sep.visualization import Visualizer

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

    def __init__(
        self,
        config: AppConfig | ValidatedConfig,
        *,
        runtime_resolver: RuntimeResolver | None = None,
    ) -> None:
        self._config = to_validated_config(config)
        self._runtime_resolver = (
            runtime_resolver if runtime_resolver is not None else TorchRuntimeResolver()
        )
        self._separation_engine: Optional[SeparationEngine] = None
        self._last_resolved_config: ResolvedConfig | None = None

    @property
    def resolved_config(self) -> ResolvedConfig | None:
        """返回最近一次完整的不可变设备解析快照。"""

        return self._last_resolved_config

    def _resolve_separation_config(self) -> tuple[RuntimeResolution, AppConfig]:
        separation_resolution = resolve_device(
            self._config.separation_device,
            backend="torch",
            resolver=self._runtime_resolver,
        )
        run_config = to_app_config(
            self._config,
            separation_device=separation_resolution.actual,
        )
        return separation_resolution, run_config

    def run(self, input_file: Path, dry_run: bool = False) -> PipelineResult:
        """执行完整流水线。"""

        self._last_resolved_config = None
        logger.info("=" * 50)
        logger.info("Stage 1: 校验与准备")
        logger.info("=" * 50)

        validated_path = validate_input_file(input_file)
        logger.info("输入文件: %s", validated_path)

        separation_resolution, run_config = self._resolve_separation_config()
        logger.info("分离设备: %s", separation_resolution.actual)
        if not self._config.lyrics_enabled:
            self._last_resolved_config = ResolvedConfig(
                requested=self._config,
                separation=separation_resolution,
                lyrics=None,
            )

        self._separation_engine = SeparationEngine(run_config.separation)
        self._separation_engine.validate_output(
            run_config.output.format,
            run_config.output.bitrate,
        )

        try:
            output_paths = resolve_output_paths(
                validated_path,
                run_config.output.output_dir,
                lyrics_format=run_config.lyrics.output_format,
                overwrite=run_config.output.overwrite,
            )
        except OutputExistsError as exc:
            logger.warning(str(exc))
            raise

        if dry_run:
            self._print_dry_run_plan(validated_path, output_paths, run_config)
            return PipelineResult(
                input_file=validated_path,
                output_paths=output_paths,
            )

        ensure_output_dirs(output_paths, overwrite=run_config.output.overwrite)
        logger.info("输出目录: %s", output_paths.base_dir)

        logger.info("=" * 50)
        logger.info("Stage 2: 音乐分离")
        logger.info("=" * 50)

        result = PipelineResult(
            input_file=validated_path,
            output_paths=output_paths,
        )

        stems = self._separation_engine.separate(
            validated_path,
            output_paths,
            output_format=run_config.output.format,
            bitrate=run_config.output.bitrate,
        )
        result.stems = stems

        if run_config.lyrics.enabled:
            logger.info("=" * 50)
            logger.info("Stage 3: 歌词识别")
            logger.info("=" * 50)
            try:
                lyrics_resolution = resolve_device(
                    self._config.whisper_device,
                    backend="ctranslate2",
                    resolver=self._runtime_resolver,
                )
                self._last_resolved_config = ResolvedConfig(
                    requested=self._config,
                    separation=separation_resolution,
                    lyrics=lyrics_resolution,
                )
                lyrics_config = to_app_config(
                    self._config,
                    separation_device=separation_resolution.actual,
                    whisper_device=lyrics_resolution.actual,
                ).lyrics
                transcriber = LyricsTranscriber(
                    lyrics_config,
                    runtime_resolution=lyrics_resolution,
                )
                audio_source = result.stems.get("vocals", validated_path)
                if "vocals" in result.stems:
                    logger.info("使用分离出的人声轨进行歌词识别")
                else:
                    logger.info("未找到人声轨，使用原始音频进行歌词识别")

                transcription = transcriber.transcribe(audio_source)
                lyrics_path = transcriber.save_lyrics(
                    transcription,
                    output_paths.lyrics_path,
                    fmt=run_config.lyrics.output_format,
                )
                result.lyrics_path = lyrics_path
            except Exception as exc:
                logger.error("歌词识别失败: %s", exc)
                logger.info("跳过歌词识别，继续后续处理")

        if run_config.visualization.enabled:
            logger.info("=" * 50)
            logger.info("Stage 4: 可视化")
            logger.info("=" * 50)
            try:
                visualizer = Visualizer(run_config.visualization)
                viz_paths: list[Path] = []

                original_viz = visualizer.generate(
                    validated_path,
                    output_paths.viz_dir,
                    stem_name=None,
                )
                viz_paths.extend(original_viz)

                for stem_name, stem_path in result.stems.items():
                    stem_viz = visualizer.generate(
                        stem_path,
                        output_paths.viz_dir,
                        stem_name=stem_name,
                    )
                    viz_paths.extend(stem_viz)

                result.visualization_paths = viz_paths
            except Exception as exc:
                logger.error("可视化生成失败: %s", exc)
                logger.info("跳过可视化")

        logger.info("=" * 50)
        logger.info("处理完成!")
        logger.info("=" * 50)
        self._print_summary(result)

        return result

    def _print_dry_run_plan(
        self,
        input_file: Path,
        output_paths: OutputPaths,
        config: AppConfig,
    ) -> None:
        """打印 dry-run 执行计划。"""

        sep = config.separation
        out = config.output
        lyrics = config.lyrics
        visualization = config.visualization

        print("\n📋 执行计划 (dry-run)")
        print("=" * 50)
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_paths.base_dir}")
        print()
        print("[分离]")
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

        if lyrics.enabled:
            print()
            print("[歌词识别]")
            print(f"  Whisper 模型: {lyrics.whisper_model}")
            print(f"  设备: {lyrics.whisper_device}")
            if lyrics.language:
                print(f"  语言: {lyrics.language}")
            print(f"  输出格式: {lyrics.output_format}")

        if visualization.enabled:
            print()
            print("[可视化]")
            print(f"  类型: {', '.join(visualization.types)}")

        print("=" * 50)
        print("（dry-run 模式，未执行任何操作）\n")

    def _print_summary(self, result: PipelineResult) -> None:
        """打印处理结果摘要。"""

        print(f"\n输出目录: {result.output_paths.base_dir}")
        print(f"分离音轨: {', '.join(result.stems.keys())}")
        if result.lyrics_path:
            print(f"歌词文件: {result.lyrics_path.name}")
        if result.visualization_paths:
            print(f"可视化图片: {len(result.visualization_paths)} 张")
