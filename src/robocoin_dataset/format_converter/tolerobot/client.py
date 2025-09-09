import logging
import time
from pathlib import Path

from tqdm import tqdm

from robocoin_dataset.distribution_computation.constant import (
    DATASET_PATH,
    DEVICE_MODEL,
)
from robocoin_dataset.distribution_computation.task_client import TaskClient
from robocoin_dataset.format_converter.tolerobot.constant import (
    CONVERTER_CLASS_NAME,
    CONVERTER_CONFIG,
    CONVERTER_LOG_DIR,
    CONVERTER_MODULE_PATH,
    IMAGE_WRITER_PROCESSES,
    IMAGE_WRITER_THREADS,
    LEFORMAT_PATH,
    REPO_ID,
    VIDEO_BACKEND,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverterFactory,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter_h5 import (
    LerobotFormatConverterHdf5,
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

    def _sync_process_task(self, task_content: dict) -> None:
        try:
            dataset_path = Path(task_content.get(DATASET_PATH))
            device_model = task_content.get(DEVICE_MODEL)
            output_path = Path(task_content.get(LEFORMAT_PATH))
            converter_config = task_content.get(CONVERTER_CONFIG)
            repo_id = task_content.get(REPO_ID)
            device_model = task_content.get(DEVICE_MODEL, None)
            module_path = task_content.get(CONVERTER_MODULE_PATH)
            class_name = task_content.get(CONVERTER_CLASS_NAME)
            video_backend = task_content.get(VIDEO_BACKEND)
            image_writer_proecesses = task_content.get(IMAGE_WRITER_PROCESSES, 4)
            image_writer_threads = task_content.get(IMAGE_WRITER_THREADS, 4)
            converter_log_dir = task_content.get(CONVERTER_LOG_DIR)
            print(f"converter_log_dir: {converter_log_dir}")

            time.sleep(1)

            converter: LerobotFormatConverterHdf5 = LerobotFormatConverterFactory.create_converter(
                dataset_path=dataset_path,
                device_model=device_model,
                output_path=output_path,
                converter_config=converter_config,
                converter_module_path=module_path,
                converter_class_name=class_name,
                repo_id=repo_id,
                converter_log_dir=str(converter_log_dir),
                video_backend=video_backend,
                image_writer_processes=image_writer_proecesses,
                image_writer_threads=image_writer_threads,
            )

            total_episodes = converter.get_episodes_num()
            for task_content, task_ep_idx, ep_idx in tqdm(
                converter.convert(),
                total=total_episodes,
                desc="Converting Dataset",
                unit="episode",
            ):
                self.logger.info(
                    f"Converted episode {task_ep_idx} of task {task_content}, total ep_idx is:{ep_idx}"
                )

        except Exception as e:
            raise RuntimeError(f"convert dataset {dataset_path} failed: {e}")
