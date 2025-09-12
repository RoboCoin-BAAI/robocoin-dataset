import logging
import threading
from pathlib import Path

from robocoin_dataset.annotation.subtask_annotion.constant import (
    DS_API_KEY,
    SUBTASK_ANNOTATION_SOURCE_FILE_NAME,
    SUBTASK_ANNOTATION_SOURCE_FILE_PATH,
    SUBTASK_ANNOTATION_TARGET_FILE_PATH,
)
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    LeFormatConvertDB,
    SubtaskAnnotationDB,
    TaskStatus,
)
from robocoin_dataset.database.services.subtask_annotation import (
    upsert_subtask_annotation_status,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
    ERR_MSG,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer


class SubtaskAnnotationTaskServer(TaskServer):
    def __init__(
        self,
        db_file: Path,
        ds_api_key: str,
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
        self.ds_api_key = ds_api_key

    def get_task_category(self) -> str:
        return "subtask_annotation"

    def generate_task_content(self) -> dict | None:
        with self._db_lock:
            with self.db.with_session() as session:
                results = (
                    session.query(LeFormatConvertDB)
                    .filter(LeFormatConvertDB.convert_status == TaskStatus.COMPLETED)
                    .filter(
                        ~session.query(SubtaskAnnotationDB)
                        .filter(LeFormatConvertDB.dataset_uuid == SubtaskAnnotationDB.dataset_uuid)
                        .exists()
                    )
                    .all()
                )

            if not results:
                return None

            for item in results:
                leformat_items = session.query(LeFormatConvertDB).filter(
                    LeFormatConvertDB.dataset_uuid == item.dataset_uuid
                )
                leformat_item = leformat_items.first()
                if not leformat_item:
                    raise Exception(f"{item.dataset_uuid} in LeformatConvertDB not exists.")

                leformat_path = leformat_item.convert_path

                subtask_annotation_source_file_path = str(
                    Path(leformat_path) / SUBTASK_ANNOTATION_SOURCE_FILE_NAME
                )

                upsert_subtask_annotation_status(
                    ds_uuid=item.dataset_uuid,
                    session=session,
                )

                return {
                    DATASET_UUID: item.dataset_uuid,
                    SUBTASK_ANNOTATION_SOURCE_FILE_PATH: subtask_annotation_source_file_path,
                    DS_API_KEY: self.ds_api_key,
                }

        return None

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        print(task_content)
        print(task_result_content)
        ds_uuid = task_content.get(DATASET_UUID)

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        err_msg = task_result_content.get(ERR_MSG)
        subtask_annotation_target_file_path = task_result_content[TASK_RESULT_CONTENT][
            SUBTASK_ANNOTATION_TARGET_FILE_PATH
        ]

        print(subtask_annotation_target_file_path)

        task_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        with self.db.with_session() as session:
            upsert_subtask_annotation_status(
                session=session,
                ds_uuid=ds_uuid,
                subtask_annotation_file_path=subtask_annotation_target_file_path,
                subtask_annotation_status=task_status,
                err_msg=err_msg,
            )
            self.logger.info(
                f"Upsert {ds_uuid} subtask annotation status to {task_status}, update_message: {err_msg}"
            )
