import argparse
from typing import Union

import av
import numpy as np
import torch


def extract_all_frames_pyav(
    video_path: str,
    return_tensors: str = "np",  # "np" or "pt"
    target_size: tuple = None,  # (width, height)，可选缩放
    as_numpy_type: np.dtype = np.uint8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    使用 PyAV 提取视频中的所有帧。

    Args:
        video_path: 视频路径
        return_tensors: 返回格式 "np" 或 "pt"
        target_size: 可选，缩放帧尺寸 (width, height)
        as_numpy_type: 输出数据类型，默认 uint8

    Returns:
        shape: (T, H, W, 3)  if "np"
        shape: (T, 3, H, W)  if "pt"
    """
    frames = []

    try:
        try:
            container = av.open(video_path)
        except Exception as e:
            print(f"Error opening video {video_path}: {str(e)}")
            raise e
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # 启用多线程解码
        frame_idx = 0

        for frame in container.decode(stream):
            # 转为 RGB 并转为 numpy
            try:
                img = frame.to_rgb(
                    width=target_size[0] if target_size else None,
                    height=target_size[1] if target_size else None,
                )
                frame_np = img.to_ndarray()
                frames.append(frame_np)

            except Exception as e:
                print(f"Error processing frame {frame_idx} in {video_path}: {str(e)}")
            finally:
                frame_idx += 1

        container.close()
    except Exception as e:
        print(f"Error opening video {video_path}: {str(e)}")

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    # 堆叠成数组
    video = np.stack(frames, axis=0)  # (T, H, W, 3)

    if as_numpy_type != np.uint8:
        video = video.astype(as_numpy_type)

    if return_tensors == "np":
        return video
    if return_tensors == "pt":
        return torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
    raise ValueError("return_tensors must be 'np' or 'pt'")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Decode video file.")
    argparser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the directory of video file.",
    )

    args = argparser.parse_args()

    extract_all_frames_pyav(args.video_path, return_tensors="pt")

"""
python scripts/format_converters/tolerobot/check_single_video_pyav.py --video-path 
"""
