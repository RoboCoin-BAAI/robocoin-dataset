import logging
import threading
from pathlib import Path

from robocoin_dataset.annotation.constant import (
    DS_API_KEY,
    SUBTASK_ANNOTATION_FILE_NAME,
    SUBTASK_ANNOTATION_FILE_PATH,
)
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    LeFormatConvertDB,
    SubtaskAnnotationStatusDB,
    TaskStatus,
)
from robocoin_dataset.database.services.subtask_annotation import (
    upsert_subtask_annotation_status,
)
from robocoin_dataset.distribution_computation.constant import (
    DATASET_UUID,
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
        print("generate_task_content")
        with self._db_lock:
            with self.db.with_session() as session:
                results = (
                    session.query(LeFormatConvertDB)
                    .filter(LeFormatConvertDB.convert_status == TaskStatus.COMPLETED)
                    .filter(
                        ~session.query(SubtaskAnnotationStatusDB)
                        .filter(
                            LeFormatConvertDB.dataset_uuid == SubtaskAnnotationStatusDB.dataset_uuid
                        )
                        .exists()
                    )
                    .all()
                )
            print(f"{len(results)} results found")

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

                subtask_annotation_file_path = str(
                    Path(leformat_path) / SUBTASK_ANNOTATION_FILE_NAME
                )

                upsert_subtask_annotation_status(
                    ds_uuid=item.dataset_uuid,
                    session=session,
                    subtask_annotation_file_path=subtask_annotation_file_path,
                )

                return {
                    DATASET_UUID: item.dataset_uuid,
                    SUBTASK_ANNOTATION_FILE_PATH: subtask_annotation_file_path,
                    DS_API_KEY: self.ds_api_key,
                }

        return None
