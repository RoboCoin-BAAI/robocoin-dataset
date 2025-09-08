import argparse
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
    argparser = argparse.ArgumentParser(description="Decode video file.")
    argparser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the video file.",
    )

    args = argparser.parse_args()

    decode_all_video_frames_torchcodec(args.video_path)

"""_summary_
python scripts/format_converters/tolerobot/check_single_video_torchcodec.py --video-path
"""
