import logging

from torchvision.io import VideoReader


def decode_all_video_frames_videoreader(
    video_path: str,
    log_progress: bool = False,
) -> tuple:
    """
    使用 VideoReader 读取所有帧（兼容旧版 torchvision）
    """
    try:
        # 移除 device 参数
        vr = VideoReader(video_path, stream="video")
    except Exception as e:
        logging.error(f"❌ 无法打开视频: {video_path}, 错误: {e}")
        return None, None

    frames = []
    timestamps = []
    frame_idx = 0

    try:
        for frame_data in vr:
            try:
                pts = frame_data.pts
                img_tensor = frame_data.video  # (C, H, W), uint8

                # 归一化
                img_float = img_tensor.float().div(255.0)

                frames.append(img_float)
                timestamps.append(pts)

                if log_progress and frame_idx % 100 == 0:
                    logging.info(f"✅ 解码帧 {frame_idx}, PTS={pts:.4f}s")

                frame_idx += 1

            except Exception as e:
                logging.error(f"❌ 处理帧 {frame_idx} 失败: {e}")
                break

    except Exception as e:
        logging.error(f"🔥 解码过程中出错: {e}")
        return None, None

    if not frames:
        logging.error(f"❌ 未解码出任何帧: {video_path}")
        return None, None

    try:
        import torch

        tensor_frames = torch.stack(frames)  # (N, C, H, W)
        ts_tensor = torch.tensor(timestamps)
    except Exception as e:
        logging.error(f"❌ 张量合并失败: {e}")
        return None, None

    if log_progress:
        logging.info(f"✅ 成功解码 {len(frames)} 帧")

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

    frames, timestamps = decode_all_video_frames_videoreader(
        video_path=args.video_path,
        log_progress=args.log_progress,
    )
