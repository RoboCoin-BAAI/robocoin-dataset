import argparse
import asyncio
import logging

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
        default="mnt/nas/robocoin_dataset",
        help="Path to save lerobot converted dataset.",
    )

    args = argparser.parse_args()
    server = LeFormatConverterTaskServer(
        db_file=args.db_file,
        convert_root_path=args.convert_root_path,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        heartbeat_interval=args.heartbeat_interval,
        logger=setup_logger(
            name="leformat_convertor_server",
            log_dir=args.log_path,
            level=logging.INFO,
        ),
    )
    await server.start()  # 创建任务并调度执行


if __name__ == "__main__":
    asyncio.run(main())


"""_summary_
python scripts/format_converters/tolerobot/server.py \
    --db-file=db/datasets.db \
    --host=0.0.0.0 \
    --port=8765 \
    --timeout=1.0 \
    --heartbeat-interval=10.0 \
    --log-path=./output/leformat_converter/log
"""
