"""music-separation CLI 命令定义

基于 Typer 的命令行界面，提供 separate / models / info 三个命令。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from music_sep import __version__
from music_sep.config import merge_config
from music_sep.constants import AVAILABLE_MODELS
from music_sep.outputs import OutputExistsError
from music_sep.pipeline import Pipeline
from music_sep.utils import setup_logging

app = typer.Typer(
    name="music-sep",
    help="Music Separation - 基于 Demucs 的音轨分离与歌词识别工具",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"music-sep {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="显示版本号并退出",
    ),
) -> None:
    """Music Separation - 基于 Demucs 的音轨分离与歌词识别工具"""
    pass


# ---------------------------------------------------------------------------
# separate 命令
# ---------------------------------------------------------------------------

@app.command()
def separate(
    input: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="输入音频文件路径",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Demucs 模型名称 (默认: htdemucs)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="分离设备: cpu / cuda / mps / auto (默认: auto)",
    ),
    shifts: Optional[int] = typer.Option(
        None,
        "--shifts",
        help="随机偏移次数（越多质量越高，速度越慢，默认: 1）",
    ),
    overlap: Optional[float] = typer.Option(
        None,
        "--overlap",
        help="偏移重叠比例 (0.0 ~ 1.0，默认: 0.25)",
    ),
    two_stems: Optional[str] = typer.Option(
        None,
        "--two-stems",
        help="仅输出指定音轨 + 残差，如 --two-stems vocals",
    ),
    stems: Optional[list[str]] = typer.Option(
        None,
        "--stems",
        help="仅输出指定音轨，可多次使用，如 --stems drums --stems bass",
    ),
    lyrics: Optional[bool] = typer.Option(
        None,
        "--lyrics/--no-lyrics",
        help="启用/禁用歌词识别（默认禁用）",
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="Whisper 模型 (默认: medium)",
    ),
    whisper_device: Optional[str] = typer.Option(
        None,
        "--whisper-device",
        help="歌词识别设备: cpu / cuda / auto (默认: auto)",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="强制指定音频语言（如 zh, en, ja），不指定则自动检测",
    ),
    lyrics_format: Optional[str] = typer.Option(
        None,
        "--lyrics-format",
        help="歌词输出格式: srt / vtt / txt / json (默认: srt)",
    ),
    visualize: Optional[bool] = typer.Option(
        None,
        "--visualize/--no-visualize",
        help="启用/禁用可视化（默认禁用）",
    ),
    viz_types: Optional[list[str]] = typer.Option(
        None,
        "--viz-types",
        help="可视化类型，可多次使用: waveform / spectrogram / mel",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="输出目录 (默认: demo)",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        help="输出音频格式: wav / mp3 / flac (默认: wav)",
    ),
    bitrate: Optional[str] = typer.Option(
        None,
        "--bitrate",
        help="MP3 比特率，如 128k / 192k / 320k (默认: 128k)",
    ),
    overwrite: Optional[bool] = typer.Option(
        None,
        "--overwrite/--no-overwrite",
        help="覆盖已有输出目录 (默认: 否)",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="详细日志输出",
    ),
    quiet: bool = typer.Option(
        False,
        "-q",
        "--quiet",
        help="静默模式，仅输出错误",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="TOML 配置文件路径",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="仅显示执行计划，不实际运行",
    ),
) -> None:
    """分离音频文件中的各个音轨

    使用 Demucs 模型将音频分离为 vocals / drums / bass / other 等音轨，
    可选启用歌词识别（Whisper）和频谱可视化。
    """
    # 1. 配置日志
    logger = setup_logging(verbose=verbose, quiet=quiet)

    # 2. 三层配置合并：CLI > TOML > 默认值
    try:
        app_config = merge_config(
            toml_path=config,
            model=model,
            device=device,
            shifts=shifts,
            overlap=overlap,
            two_stems=two_stems,
            stems=stems,
            lyrics_enabled=lyrics,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            language=language,
            lyrics_format=lyrics_format,
            visualize_enabled=visualize,
            viz_types=viz_types,
            output_dir=output_dir,
            output_format=format,
            bitrate=bitrate,
            overwrite=overwrite,
        )
    except Exception as e:
        logger.error(f"配置错误: {e}")
        raise typer.Exit(code=1)

    # 3. 打印执行计划
    sep = app_config.separation
    lyr = app_config.lyrics
    viz = app_config.visualization
    out = app_config.output

    typer.echo("=" * 60)
    typer.echo("  Music Separation 执行计划")
    typer.echo("=" * 60)
    typer.echo(f"  输入文件  : {input.resolve()}")
    typer.echo(f"  输出目录  : {out.output_dir.resolve()}")
    typer.echo()
    typer.echo(f"  [分离]")
    typer.echo(f"    模型    : {sep.model}")
    typer.echo(f"    设备    : {sep.device}")
    typer.echo(f"    shifts  : {sep.shifts}")
    typer.echo(f"    overlap : {sep.overlap}")
    if sep.two_stems:
        typer.echo(f"    双轨    : {sep.two_stems}")
    if sep.stems:
        typer.echo(f"    指定轨  : {', '.join(sep.stems)}")
    typer.echo()
    typer.echo(f"  [歌词识别]")
    typer.echo(f"    启用    : {'是' if lyr.enabled else '否'}")
    if lyr.enabled:
        typer.echo(f"    Whisper : {lyr.whisper_model}")
        typer.echo(f"    设备    : {lyr.whisper_device}")
        typer.echo(f"    语言    : {lyr.language or '自动检测'}")
        typer.echo(f"    格式    : {lyr.output_format}")
    typer.echo()
    typer.echo(f"  [可视化]")
    typer.echo(f"    启用    : {'是' if viz.enabled else '否'}")
    if viz.enabled:
        typer.echo(f"    类型    : {', '.join(viz.types)}")
    typer.echo()
    typer.echo(f"  [输出]")
    typer.echo(f"    格式    : {out.format}")
    typer.echo(f"    比特率  : {out.bitrate}")
    typer.echo(f"    覆盖    : {'是' if out.overwrite else '否'}")
    typer.echo("=" * 60)

    if dry_run:
        typer.echo("\n[dry-run] 仅显示计划，退出。")
        raise typer.Exit()

    # 4. 实际 pipeline 调用
    logger.info("开始音轨分离 ...")
    try:
        pipeline = Pipeline(app_config)
        result = pipeline.run(input, dry_run=dry_run)
    except OutputExistsError as e:
        logger.warning(str(e))
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        raise typer.Exit(code=1)
    logger.info("完成。")


# ---------------------------------------------------------------------------
# models 命令
# ---------------------------------------------------------------------------

@app.command()
def models(
    installed_only: bool = typer.Option(
        False,
        "--installed-only",
        help="仅列出已安装的模型",
    ),
) -> None:
    """列出可用的 Demucs 模型

    显示模型名称、描述以及是否已缓存在本地。
    模型缓存目录: ~/.cache/torch/hub/
    """
    cache_dir = Path.home() / ".cache" / "torch" / "hub"

    # 扫描已有的 demucs 模型缓存
    installed_cache_names: set[str] = set()
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_dir() and "demucs" in item.name.lower():
                installed_cache_names.add(item.name)

    # 渲染模型列表
    typer.echo("可用 Demucs 模型:")
    typer.echo("-" * 70)

    has_installed = False
    for name, description in AVAILABLE_MODELS.items():
        # 简单判断是否已安装：缓存目录名中包含模型名
        is_installed = any(name in cached for cached in installed_cache_names)

        if installed_only and not is_installed:
            continue

        status = typer.style("[已安装]", fg=typer.colors.GREEN) if is_installed else typer.style("[未安装]", fg=typer.colors.YELLOW)
        typer.echo(f"  {name:<16s} {status}  {description}")

        if is_installed:
            has_installed = True

    if installed_only and not has_installed:
        typer.echo("  (没有已安装的模型，首次使用时会自动下载)")
    elif not has_installed:
        typer.echo()
        typer.echo("  提示: 模型将在首次使用时自动下载。")

    typer.echo("-" * 70)
    typer.echo(f"共 {len(AVAILABLE_MODELS)} 个模型")


# ---------------------------------------------------------------------------
# info 命令
# ---------------------------------------------------------------------------

@app.command()
def info(
    input: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="音频文件路径",
    ),
) -> None:
    """查看音频文件元信息

    显示文件名、格式、时长、采样率、声道数和文件大小。
    """
    import librosa
    import soundfile as sf

    from music_sep.utils import format_duration

    file_path = Path(input).resolve()

    # 使用 librosa 获取采样率和时长
    y, sr = librosa.load(str(file_path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)

    # 使用 soundfile 获取声道数（更可靠）
    sf_info = sf.info(str(file_path))
    channels = sf_info.channels

    # 文件大小
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    typer.echo(f"文件名  : {file_path.name}")
    typer.echo(f"格式    : {file_path.suffix.lstrip('.').upper()}")
    typer.echo(f"时长    : {format_duration(duration)}")
    typer.echo(f"采样率  : {sr} Hz")
    typer.echo(f"声道数  : {channels} ({'立体声' if channels == 2 else '单声道' if channels == 1 else f'{channels}声道'})")
    typer.echo(f"文件大小: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    app()
