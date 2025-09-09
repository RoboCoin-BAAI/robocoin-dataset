# video_corruption_checker.py
import argparse
from pathlib import Path

from tqdm import tqdm


# =============================
# Backend: PyAV
# =============================
def _check_video_pyav(video_path):
    import av

    errors = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_count = 1  # 单线程更稳定解码错误捕获

        for frame in container.decode(video=0):
            try:
                frame.to_rgb()  # 触发解码
            except Exception as e:
                errors.append(f"Bad frame at time={frame.time:.3f}s: {e}")
        container.close()
    except Exception as e:
        errors.append(f"Failed to open/decode: {e}")
    return len(errors) == 0, errors


# =============================
# Backend: torchcodec
# =============================
def _check_video_torchcodec(video_path: Path | str) -> tuple[bool, list[str]]:
    """
    使用 torchcodec 检查视频中的每一帧是否可正常解码。

    Args:
        video_path: 视频文件路径

    Returns:
        (is_valid, errors):
            is_valid: True 表示所有帧解码成功，False 表示存在错误
            errors: 错误信息列表，若为空则表示无错误
    """
    import importlib
    import logging
    from typing import List

    errors: List[str] = []

    # 检查 torchcodec 是否可用
    if importlib.util.find_spec("torchcodec") is None:
        errors.append("torchcodec 未安装。请安装: pip install torchcodec")
        return False, errors

    try:
        from torchcodec.decoders import VideoDecoder
    except Exception as e:
        errors.append(f"Failed to import torchcodec: {e}")
        return False, errors

    video_path_str = str(video_path)

    try:
        # 初始化解码器
        decoder = VideoDecoder(video_path_str, device="cpu", seek_mode="approximate")
        metadata = decoder.metadata

        num_frames = metadata.num_frames
        average_fps = metadata.average_fps

        if num_frames <= 0:
            errors.append(f"无法获取有效帧数：{video_path_str}")
            return False, errors

        logging.info(
            f"开始检查视频帧解码: {video_path_str} ({num_frames} 帧, {average_fps:.2f} FPS)"
        )

        # 遍历每一帧，尝试解码
        for frame_idx in range(num_frames):
            try:
                frame = decoder.get_frame_at(index=frame_idx)
                # 触发实际解码（访问 tensor 数据）
                _ = frame.video[0]  # 确保帧数据被加载
            except Exception as e:
                pts_seconds = (
                    frame_idx / average_fps
                    if "frame" not in locals()
                    else getattr(frame, "pts", frame_idx / average_fps)
                )
                errors.append(f"Bad frame at index={frame_idx}, time={pts_seconds:.3f}s: {e}")

    except Exception as e:
        errors.append(f"Failed to open/decode video: {e}")
    return len(errors) == 0, errors


# =============================
# Unified Scanner Function
# =============================
def scan_videos_for_corrupted_frames(root_dir, backend="pyav"):
    """
    遍历目录（含子目录），检查所有视频文件中的坏帧。

    参数:
        root_dir (str or Path): 视频根目录
        backend (str): 后端解码器，可选 'pyav' 或 'torchcodec'

    返回:
        bad_videos: List[Tuple[str, List[str]]]，包含损坏视频路径和错误信息
    """
    if backend not in ["pyav", "torchcodec"]:
        raise ValueError("backend must be 'pyav' or 'torchcodec'")

    # 导入依赖（运行时检查）
    if backend == "pyav":
        try:
            import av
        except ImportError:
            raise ImportError("pyav not installed. Please run: pip install av")
    else:  # torchcodec
        try:
            import torchcodec
        except ImportError:
            raise ImportError(
                "torchcodec not installed. Please install from https://github.com/pytorch/torchcodec"
            )

    # 支持的视频扩展名
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
    root_path = Path(root_dir)

    # 获取所有视频文件
    video_files = [
        f for f in root_path.rglob("*") if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"🔍 No video files found in: {root_dir}")
        return []

    # 选择检测函数
    checker = _check_video_pyav if backend == "pyav" else _check_video_torchcodec

    bad_videos = []

    desc = f"Scanning videos ({backend})"
    with tqdm(total=len(video_files), desc=desc, unit="video") as pbar:
        for filepath in video_files:
            try:
                is_ok, errors = checker(filepath)
                if not is_ok:
                    bad_videos.append((str(filepath), errors))
            except Exception as e:
                bad_videos.append((str(filepath), [f"Unhandled error: {e}"]))
            finally:
                pbar.set_postfix_str(Path(filepath).name)  # 显示当前文件名
                pbar.update(1)

    return bad_videos


# =============================
# 使用示例
# =============================
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Check all videos in a directory for corrupted frames."
    )
    argparser.add_argument("directory", type=str, help="Directory to scan for videos.")

    argparser.add_argument(
        "--backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Backend to use for video decoding.",
    )

    print("🔍 Starting video integrity check...\n")

    args = argparser.parse_args()
    bad_videos = scan_videos_for_corrupted_frames(args.directory, backend=args.backend)

    # 输出结果
    print(f"\n❌ Found {len(bad_videos)} corrupted video(s):\n")
    for path, errors in bad_videos:
        print(f"📹 {path}")
        for err in errors:
            print(f"   ⚠️  {err}")
        print("-" * 60)
