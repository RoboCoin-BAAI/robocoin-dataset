import argparse
import asyncio
import logging

from robocoin_dataset.format_converter.tolerobot.client import LeFormatConverterTaskClient
from robocoin_dataset.utils.logger import setup_logger


async def main() -> None:
    argparser = argparse.ArgumentParser(description="Convert dataset to lerobot format.")
    argparser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="server host to connect to.",
    )

    argparser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="server port to connect to.",
    )

    argparser.add_argument(
        "--log-path",
        type=str,
        default="logs/client.log",
        help="Path to log file.",
    )

    argparser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Timeout for each client.",
    )

    argparser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="Heartbeat interval for each client.",
    )

    args = argparser.parse_args()

    server_uri = f"ws://{args.host}:{args.port}"

    client = LeFormatConverterTaskClient(
        server_uri=server_uri,
        heartbeat_interval=args.heartbeat_interval,
        logger=setup_logger(
            name="leformat_convertor_client",
            log_dir=args.log_path,
            level=logging.INFO,
        ),
    )
    await client.run()  # 创建任务并调度执行


if __name__ == "__main__":
    asyncio.run(main())


"""_summary_
python scripts/format_converters/tolerobot/client.py \
    --host=127.0.0.1 \
    --port=8765 \
    --timeout=1.0 \
    --heartbeat-interval=10.0 \
    --log-path=./output/leformat_converter/log
"""
