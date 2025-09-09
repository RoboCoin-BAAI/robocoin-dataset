import argparse
import asyncio
import logging
import multiprocessing as mp

from robocoin_dataset.format_converter.tolerobot.client import LeFormatConverterTaskClient
from robocoin_dataset.utils.logger import setup_logger


async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """
    每个进程运行的异步客户端逻辑。
    """
    # 为每个进程创建独立的日志文件或使用共享日志但区分进程
    logger = setup_logger(
        name=f"client_{process_id}",
        log_dir=log_path,
        level=logging.ERROR,
    )

    client = LeFormatConverterTaskClient(
        server_uri=server_uri,
        heartbeat_interval=heartbeat_interval,
        logger=logger,
    )
    await client.run()


def client_process_main(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """
    多进程入口函数，每个进程启动自己的 asyncio 事件循环。
    """
    asyncio.run(
        run_client_process(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            log_path=log_path,
            process_id=process_id,
        )
    )


def main() -> None:
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
    argparser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of concurrent client processes to spawn.",
    )

    args = argparser.parse_args()

    num_clients = max(1, min(args.num_clients, 8))

    server_uri = f"ws://{args.host}:{args.port}"

    # 使用 multiprocessing 启动多个客户端进程
    processes = []
    for i in range(num_clients):
        proc = mp.Process(
            target=client_process_main,
            kwargs=dict(
                server_uri=server_uri,
                heartbeat_interval=args.heartbeat_interval,
                log_path=args.log_path,
                process_id=i,
            ),
        )
        proc.start()
        processes.append(proc)

    print(f"Started {num_clients} client processes. Waiting for them to finish...")

    try:
        for proc in processes:
            proc.join()  # 等待所有进程结束
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)


if __name__ == "__main__":
    # Windows 兼容性：避免多进程重复执行入口
    mp.set_start_method("spawn", force=True)
    main()


"""_summary_
python scripts/format_converters/tolerobot/multi_client.py \
    --host=127.0.0.1 \
    --port=8765 \
    --timeout=1.0 \
    --heartbeat-interval=10.0 \
    --log-path=./output/leformat_converter/log \
    --num-clients=4
"""
