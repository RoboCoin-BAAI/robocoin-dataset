import logging
from pathlib import Path
from typing import Optional, Tuple

import av
import numpy as np
import torch


def decode_all_video_frames_pyav(
    video_path: Path | str,
    device: str = "cpu",
    log_progress: bool = False,
    return_tensors: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    使用 PyAV 读取视频中的所有帧，并返回帧张量和时间戳。

    Args:
        video_path: 视频文件路径
        device: 输出张量所在设备（仅在 return_tensors=True 时有效）
        log_progress: 是否打印进度
        return_tensors: 是否返回 torch.Tensor；若 False，则返回 ndarray 列表

    Returns:
        frames: 形状为 (N, C, H, W) 的张量，值在 [0,1] 范围内，dtype=torch.float32
                或 None（如果解码失败）
        timestamps: 形状为 (N,) 的张量，包含每帧的 PTS（秒），单位：秒

    注意：
        - PyAV 对某些编码异常更敏感（如 B-frame 错误、NAL 损坏），适合做“质检”
        - 若某帧解码失败，会跳过并记录警告
    """
    video_path = str(video_path)
    print(video_path)
    try:
        container = av.open(video_path)
    except Exception as e:
        logging.error(f"❌ 无法打开视频文件: {video_path}, 错误: {e}")
        return None, None

    # 获取视频流（第一个视频流）
    video_stream = None
    for stream in container.streams:
        if stream.type == "video":
            video_stream = stream
            break

    if not video_stream:
        logging.error(f"❌ 视频中未找到视频流: {video_path}")
        container.close()
        return None, None

    # 设置解码器参数（提高容错性）
    video_stream.thread_type = "AUTO"
    # 可选：设置最大分析时长（避免卡住）
    # container.streams.video[0].codec_context.skip_frame = 'NONKEY'  # 跳过非关键帧（可选）

    frames = []
    timestamps = []

    frame_idx = 0
    try:
        for packet in container.demux(video_stream):
            try:
                for frame in packet.decode():
                    # 获取帧数据 (H, W, C)
                    img = frame.to_ndarray(format="rgb24")

                    # 获取时间戳（秒）
                    pts = frame.pts
                    time_base = video_stream.time_base
                    timestamp_sec = (
                        float(pts * time_base)
                        if pts is not None
                        else float(frame_idx / video_stream.average_rate)
                    )

                    frames.append(img)
                    timestamps.append(timestamp_sec)

                    if log_progress and frame_idx % 100 == 0:
                        logging.info(f"✅ 解码帧 {frame_idx}, PTS={timestamp_sec:.4f}s")

            except Exception as e:
                logging.warning(f"⚠️ 解码 packet 时出错（跳过）: 帧 {frame_idx}, 错误: {e}")
            finally:
                frame_idx += 1

    except Exception as e:
        logging.error(f"🔥 解码过程中发生致命错误: {e}")
    finally:
        container.close()

    if len(frames) == 0:
        logging.error(f"❌ 未解码出任何帧: {video_path}")
        return None, None

    if not return_tensors:
        return frames, np.array(timestamps, dtype=np.float32)

    # 转换为 torch.Tensor
    try:
        tensor_frames = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1) for img in frames]
        )  # (N, C, H, W)
        tensor_frames = tensor_frames.float() / 255.0
        tensor_frames = tensor_frames.to(device)
        ts_tensor = torch.tensor(timestamps, dtype=torch.float32).to(device)
    except Exception as e:
        logging.error(f"❌ 转换为 torch.Tensor 失败: {e}")
        return None, None

    if log_progress:
        logging.info(f"✅ 成功解码 {len(frames)} 帧 from {video_path}")

    return tensor_frames, ts_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Decode all video frames using torchcodec.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for decoding (cpu or cuda)."
    )
    parser.add_argument("--log_progress", action="store_true", help="Log progress during decoding.")
    args = parser.parse_args()

    frames, timestamps = decode_all_video_frames_pyav(
        video_path=args.video_path,
        device=args.device,
        log_progress=args.log_progress,
    )
