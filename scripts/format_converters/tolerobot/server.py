import argparse
import asyncio
import logging
from pathlib import Path

from robocoin_dataset.format_converter.tolerobot.server import LeFormatConverterTaskServer
from robocoin_dataset.utils.logger import setup_logger


async def main() -> None:
    argparser = argparse.ArgumentParser(description="Convert dataset to lerobot format.")
    argparser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to.",
    )
    argparser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to.",
    )
    argparser.add_argument(
        "--log-path",
        type=str,
        default="logs/server.log",
        help="Path to log file.",
    )
    argparser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for each client.",
    )

    argparser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=60.0,
        help="Heartbeat interval for each client.",
    )

    argparser.add_argument(
        "--db-file",
        type=str,
        default="db/dataset.db",
        help="Path to database file.",
    )

    argparser.add_argument(
        "--convert-root-path",
        type=str,
        default="/mnt/nas/robocoin_datasets",
        help="Path to save lerobot converted dataset.",
    )

    argparser.add_argument(
        "--converter-factory-config-path",
        type=str,
        default="configs/lerobot_format_converter_factory.yaml",
        help="Path to lerobot format converter factory config file.",
    )

    argparser.add_argument(
        "--specific-device-model",
        type=str,
        default=None,
        help="If specified, only process tasks for this device model.",
    )
    argparser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Video backend to use. Options: pyav, cv2. Default: pyav",
    )

    argparser.add_argument(
        "--image-writer-processes",
        type=int,
        default=4,
        help="Number of processes to use for image writing. Default: 4",
    )

    argparser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of threads per process to use for image writing. Default: 4",
    )

    args = argparser.parse_args()

    server = LeFormatConverterTaskServer(
        db_file=args.db_file,
        converter_factory_config_path=args.converter_factory_config_path,
        convert_root_path=args.convert_root_path,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        specific_device_model=args.specific_device_model,
        heartbeat_interval=args.heartbeat_interval,
        logger=setup_logger(
            name="leformat_convertor_server",
            log_dir=Path(args.log_path),
            level=logging.INFO,
        ),
        video_backend=args.video_backend,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    await server.start()  # 创建任务并调度执行


if __name__ == "__main__":
    asyncio.run(main())


""" usage:
python scripts/format_converters/tolerobot/server.py \
    --db-file=db/datasets.db \
    --host=172.16.18.160 \
    --port=8765 \
    --timeout=1.0 \
    --converter-factory-config-path=scripts/format_converters/tolerobot/configs/converter_factory_config.yaml \
    --specific-device-model=unitree_g1 \
    --heartbeat-interval=100000.0 \
    --log-path=/mnt/nas/robocoin_datasets/server_logs/ \
    --image-writer-processes=4 \
    --image-writer-threads=4 \
    --video-backend=pyav \
    --convert-root-path=/mnt/nas/robocoin_datasets


# for test
python scripts/format_converters/tolerobot/server.py \
    --db-file=/home/adminpc1/Desktop/3/datasets.db \
    --host=0.0.0.0 \
    --port=8765 \
    --timeout=10.0 \
    --converter-factory-config-path=scripts/format_converters/tolerobot/configs/converter_factory_config.yaml \
    --heartbeat-interval=100.0 \
    --log-path=/mnt/nas/robocoin_datasets_test/test/server_logs/ \
    --image-writer-processes=4 \
    --image-writer-threads=2 \
    --video-backend=pyav \
    --convert-root-path=/mnt/nas/robocoin_datasets_test/test \
    --specific-device-model=realman_rmc_aidal
"""
