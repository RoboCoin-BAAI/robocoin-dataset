import importlib
import logging
from pathlib import Path
from typing import Tuple

import torch


def decode_all_video_frames_torchcodec(
    video_path: Path | str,
    device: str = "cpu",
    log_progress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 torchcodec 读取视频中的所有帧。

    Args:
        video_path: 视频文件路径
        device: 解码设备，"cpu" 或 "cuda"（注意：在 DataLoader worker 中使用 CUDA 可能导致初始化错误）
        log_progress: 是否打印进度

    Returns:
        frames: 形状为 (N, C, H, W) 的张量，值在 [0,1] 范围内，dtype=torch.float32
        timestamps: 形状为 (N,) 的张量，包含每帧的 PTS（秒）

    Raises:
        ImportError: 如果 torchcodec 未安装
        RuntimeError: 如果视频无法打开或解码失败
    """
    if importlib.util.find_spec("torchcodec") is None:
        raise ImportError(
            "torchcodec is required but not available. Install it via: pip install torchcodec"
        )

    from torchcodec.decoders import VideoDecoder

    video_path = str(video_path)
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    metadata = decoder.metadata

    # 获取总帧数和 FPS
    num_frames = metadata.num_frames
    average_fps = metadata.average_fps

    if num_frames <= 0:
        raise RuntimeError(f"无法获取有效帧数：{video_path}")

    if log_progress:
        logging.info(f"开始解码视频: {video_path} ({num_frames} 帧, {average_fps:.2f} FPS)")

    # 获取所有帧的索引
    all_indices = list(range(num_frames))

    # 批量解码所有帧
    for frame_idx in all_indices:
        try:
            decoder.get_frame_at(index=frame_idx)  # 预热以避免首次调用延迟
        except Exception as e:
            print(f"❌ 无法解码帧 {frame_idx}，错误: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode all video frames using torchcodec.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for decoding (cpu or cuda)."
    )
    parser.add_argument("--log_progress", action="store_true", help="Log progress during decoding.")
    args = parser.parse_args()

    frames, timestamps = decode_all_video_frames_torchcodec(
        video_path=args.video_path,
        device=args.device,
        log_progress=args.log_progress,
    )

    print(f"Decoded {frames.shape[0]} frames from {args.video_path}.")
    print(f"Frame tensor shape: {frames.shape}")
    print(f"Timestamps tensor shape: {timestamps.shape}")
