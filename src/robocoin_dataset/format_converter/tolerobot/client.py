import logging
import random
import time

from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.format_converter.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
    LerobotFormatConvertorFactory,
)


class LeFormatConverterTaskClient(TaskClient):
    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            logger=logger,
        )

    def get_task_category(self) -> str:
        return "LeFormatConvert"

    def generate_task_request_desc(self) -> dict:
        """客户端可自定义任务请求参数"""
        return {}

    def _sync_process_task(self, task: dict) -> None:
        type = random.randint(0, 1)
        time.sleep(0.1)
        if type == 0:
            # raise RuntimeError("模拟任务失败")
            return
        if type == 1:
            raise RuntimeError("模拟任务失败")
        return
