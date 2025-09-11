# class SubTaskAnnotationTaskServer(TaskServer):
#     def __init__(
#         self,
#         db_file: Path,
#         ds_api_key: str,
#         host: str = "0.0.0.0",
#         port: int = 8765,
#         heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
#         timeout: float = 15.0,  # 等待 pong 超过15秒则断开
#         logger: logging.Logger | None = None,
#     ) -> None:
import argparse
import asyncio
import logging
from pathlib import Path

from robocoin_dataset.annotation.subtask_annotation_server import SubtaskAnnotationTaskServer
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
        "--ds-api-key",
        type=str,
        help="Deepseek API key.",
    )
    args = argparser.parse_args()

    server = SubtaskAnnotationTaskServer(
        db_file=args.db_file,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        heartbeat_interval=args.heartbeat_interval,
        ds_api_key=args.ds_api_key,
        logger=setup_logger(
            name="leformat_convertor_server",
            log_dir=Path(args.log_path),
            level=logging.INFO,
        ),
    )

    await server.start()  # 创建任务并调度执行


if __name__ == "__main__":
    asyncio.run(main())


""" usage:
python scripts/annotation/subtask_annotation_server.py \
    --db-file=db/datasets.db \
    --host=0.0.0.0 \
    --port=8765 \
    --timeout=1.0 \
    --heartbeat-interval=100000.0 \
    --log-path=./robocoin_datasets/server_logs/subtask_annotation/ \
    --ds-api-key=sk-a3c8736391cf43809957329f28cac287 

"""
