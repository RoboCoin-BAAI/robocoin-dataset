import logging
import random
import threading
from pathlib import Path

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import ConvertStatus, DatasetDB, LeFormatConvertDB
from robocoin_dataset.database.services.leformat_converter import upsert_leformat_convert
from robocoin_dataset.distribution_computation.constant import (
    DATASET_NAME,
    DATASET_PATH,
    DATASET_UUID,
    DEVICE_MODEL,
    TASK_RESULT_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import LEFORMAT_PATH


class LeFormatConverterTaskServer(TaskServer):
    def __init__(
        self,
        db_file: Path,
        convert_root_path: Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
            host=host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        db_file_path = Path(db_file).expanduser().absolute()
        self.db = DatasetDatabase(db_file=db_file_path)
        self._db_lock = threading.Lock()
        if not convert_root_path:
            raise ValueError("convert_root must be provided")
        self.convert_root_path = convert_root_path
        return

    def get_task_category(self) -> str:
        return "lerobot_format_convert"

    def generate_task_content(self) -> dict | None:
        with self._db_lock:
            with self.db.with_session() as session:
                results = (
                    session.query(DatasetDB)
                    .filter(
                        ~session.query(LeFormatConvertDB)
                        .filter(LeFormatConvertDB.dataset_uuid == DatasetDB.dataset_uuid)
                        .exists()
                    )
                    .all()
                )

                if not results:
                    return None

                item = random.choice(results)
                dataset_path = str(Path(item.yaml_file_path).parent)
                leformat_path = str(
                    Path(self.convert_root_path) / f"{item.device_model}_{item.dataset_name}"
                )

                # 在同一个 session 或新开一个
                upsert_leformat_convert(
                    session=session,
                    ds_uuid=item.dataset_uuid,
                    convert_status=ConvertStatus.PROCESSING,
                )

                return {
                    DATASET_UUID: item.dataset_uuid,
                    DATASET_NAME: item.dataset_name,
                    LEFORMAT_PATH: leformat_path,
                    DATASET_PATH: dataset_path,
                    DEVICE_MODEL: item.device_model,
                }

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)
        leformat_path = task_content.get(LEFORMAT_PATH, "")

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(TASK_RESULT_MSG)

        convert_status = (
            ConvertStatus.COMPLETED if task_status == TASK_SUCCESS else ConvertStatus.FAILED
        )

        with self._db_lock:
            with self.db.with_session() as session:
                upsert_leformat_convert(
                    session=session,
                    ds_uuid=ds_uuid,
                    convert_status=convert_status,
                    leformat_path=leformat_path,
                    update_message=task_status_msg,
                )
