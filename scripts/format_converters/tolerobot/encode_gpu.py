import argparse
import sys
import time
from pathlib import Path

import av
from lerobot.datasets.video_utils import (
    encode_video_frames,
    encode_video_frames_gpu,
)


def main():
    parser = argparse.ArgumentParser(
        description="Encode a sequence of PNG images into a video using GPU-accelerated encoding (e.g., h264_nvenc).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s frames/ output.mp4 --fps 30 --vcodec h264_nvenc --cq 18 --g 24 --overwrite
  %(prog)s frames/ output.mp4 --fps 60 --vcodec hevc_nvenc --cq 20 --preset p5 --log-level INFO
  %(prog)s frames/ output.mov --fps 25 --vcodec h264_videotoolbox --cq 17 --overwrite
        """,
    )

    # Required arguments
    parser.add_argument(
        "imgs_dir",
        type=Path,
        help="Input directory containing PNG image sequence (e.g., frame_000001.png, frame_000002.png)",
    )
    parser.add_argument(
        "video_path", type=Path, help="Output video file path (e.g., output.mp4, output.mov)"
    )
    parser.add_argument(
        "--fps", type=int, required=True, help="Frame rate of the output video (e.g., 24, 30, 60)"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264_nvenc",
        # choices=[
        #     "h264_nvenc",
        #     "hevc_nvenc",  # NVIDIA
        #     "h264_amf",
        #     "hevc_amf",  # AMD
        #     "h264_videotoolbox",
        #     "hevc_videotoolbox",  # Apple
        # ],
        help="GPU-accelerated video codec (default: h264_nvenc)",
    )

    parser.add_argument(
        "--pix-fmt",
        type=str,
        default="yuv420p",
        choices=["yuv420p", "yuvj420p"],
        help="Pixel format (default: yuv420p)",
    )

    parser.add_argument(
        "--g",
        type=int,
        default=24,
        help="GOP size (Group of Pictures), i.e., keyframe interval in frames (default: 24)",
    )

    parser.add_argument(
        "--cq",
        type=int,
        default=18,
        help="Constant Quality value (0-51), lower means higher quality (default: 18)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="p7",
        choices=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        help="Encoder preset: p1=fastest, p7=highest quality (default: p7)",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default="high",
        choices=["baseline", "main", "high"],
        help="H.264/HEVC profile (default: high)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="FFmpeg/libav logging level (default: ERROR)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it already exists"
    )

    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU-accelerated encoding (default: True)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Map string log level to av.logging constant
    log_level_map = {
        "DEBUG": av.logging.DEBUG,
        "INFO": av.logging.INFO,
        "WARNING": av.logging.WARNING,
        "ERROR": av.logging.ERROR,
        "CRITICAL": av.logging.CRITICAL,
    }

    try:
        # Call your encoding function
        st = time.time()
        if args.use_gpu:
            print("⚙️ Using GPU-accelerated encoding", file=sys.stderr)
            encode_video_frames_gpu(
                imgs_dir=args.imgs_dir,
                video_path=args.video_path,
                fps=args.fps,
                vcodec=args.vcodec,
                pix_fmt=args.pix_fmt,
                g=args.g,
                cq=args.cq,
                preset=args.preset,
                profile=args.profile,
                log_level=log_level_map[args.log_level],
                overwrite=args.overwrite,
            )
        else:
            print("⚙️ Using CPU encoding", file=sys.stderr)
            encode_video_frames(
                imgs_dir=args.imgs_dir,
                video_path=args.video_path,
                fps=args.fps,
                vcodec=args.vcodec,
                pix_fmt=args.pix_fmt,
                g=args.g,
                crf=args.cq,
                fast_decode=0,
                log_level=log_level_map[args.log_level],
                overwrite=args.overwrite,
            )
        print(f"✅ Success: Video saved to '{args.video_path}'", file=sys.stderr)
        print(f"⏱ Time taken: {time.time() - st:.2f} seconds", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""_summary_
python scripts/format_converters/tolerobot/encode_gpu.py \
    outputs/lerobot_converter/output_1/robocoin/realman_rmc_aidal_stir_coffee/images/observation.images.cam_high_rgb/episode_000000/ \
    outputs/lerobot_video_gpu_encoding/output_gpu_encode.mp4 \
    --fps 30 \
    --vcodec h264_nvenc \
    --cq 18 \
    --g 2 \
    --preset p7 \
    --overwrite \
    --use-gpu


python scripts/format_converters/tolerobot/encode_gpu.py \
    outputs/lerobot_converter/output_1/robocoin/realman_rmc_aidal_stir_coffee/images/observation.images.cam_high_rgb/episode_000000/ \
    outputs/lerobot_video_gpu_encoding/output_gpu_libsvtav1.mp4 \
    --fps 30 \
    --vcodec libsvtav1 \
    --cq 18 \
    --g 2 \
    --preset p1 \
    --overwrite 

    """
