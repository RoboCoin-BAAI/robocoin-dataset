import logging
from pathlib import Path

import yaml

from robocoin_dataset.constant import ROBOCOIN_PLATFORM
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    DmvAnnotationDB,
    LeFormatConvertDB,
    TaskStatus,
)
from robocoin_dataset.database.services.leformat_converter import upsert_leformat_convert
from robocoin_dataset.distribution_computation.constant import (
    DATASET_NAME,
    DATASET_PATH,
    DATASET_UUID,
    DEVICE_MODEL,
    ERR_MSG,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)
from robocoin_dataset.distribution_computation.task_server import TaskServer
from robocoin_dataset.format_converter.tolerobot.constant import (
    CONVERTER_CLASS_NAME,
    CONVERTER_CONFIG,
    CONVERTER_LOG_DIR,
    CONVERTER_LOG_NAME,
    CONVERTER_MODULE_PATH,
    DEFAULT_DEVICE_MODEL_VERSION,
    DEVICE_MODEL_VERSION_KEY,
    IMAGE_WRITER_PROCESSES,
    IMAGE_WRITER_THREADS,
    LEFORMAT_PATH,
    REPO_ID,
    VIDEO_BACKEND,
)


class LeFormatConverterTaskServer(TaskServer):
    def __init__(
        self,
        db_file: Path,
        convert_root_path: Path,
        converter_factory_config_path: Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,  # 服务端每30秒发一次 ping
        timeout: float = 15.0,  # 等待 pong 超过15秒则断开
        specific_device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
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
        if not convert_root_path:
            raise ValueError("convert_root must be provided")
        self.convert_root_path = convert_root_path
        self.specific_device_model = specific_device_model
        self.factory_config_dir = Path(converter_factory_config_path).parent
        self.video_backend = video_backend
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads

        try:
            with open(converter_factory_config_path) as f:
                self.converter_factory_config = yaml.safe_load(f)
        except Exception:
            raise
        return

    def _get_converter_class_name(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["class"]

    def _get_converter_module_path(self, device_model: str) -> str:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        return self.converter_factory_config[device_model]["module"]

    def _get_converter_config(self, device_model: str) -> dict:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")
        convertor_config_path = (
            self.factory_config_dir
            / self.converter_factory_config[device_model]["converter_config_path"]
        )
        try:
            with open(convertor_config_path) as f:
                convertor_config = yaml.safe_load(f)
        except Exception:
            raise
        return convertor_config

    def _get_converter_module_class_config(
        self, device_model: str, device_model_version: str = ""
    ) -> tuple[str, str, dict]:
        if device_model not in self.converter_factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        device_model_configs_list = self.converter_factory_config[device_model]
        if device_model_configs_list is None:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        if not isinstance(device_model_configs_list, list):
            raise ValueError(f"Device model {device_model} config must be a list.")

        converter_module_path: str | None = None
        for device_model_config in device_model_configs_list:
            if not device_model_version:
                if device_model_config[DEVICE_MODEL_VERSION_KEY] == DEFAULT_DEVICE_MODEL_VERSION:
                    converter_module_path = device_model_config["module"]
                    converter_class_name = device_model_config["class"]
                    converter_config_file_path = (
                        self.factory_config_dir / device_model_config["converter_config_path"]
                    )
                break
            if device_model_config[DEVICE_MODEL_VERSION_KEY] == device_model_version:
                converter_module_path = device_model_config["module"]
                converter_class_name = device_model_config["class"]
                converter_config_file_path = (
                    self.factory_config_dir
                    / device_model_config["converter_config_path"]
                )
                break
        
        if converter_module_path is None:
            raise ValueError(
                f"Device model {device_model} with version {device_model_version} not found in factory config."
            )
        with open(converter_config_file_path) as f:
            converter_config = yaml.safe_load(f)
        return converter_module_path, converter_class_name, converter_config

    def get_task_category(self) -> str:
        return "lerobot_format_convert"

    def generate_task_content(self) -> dict | None:
        with self.db.with_session() as session:
            if not self.specific_device_model:
                results = (
                    session.query(DmvAnnotationDB)
                    .filter(DmvAnnotationDB.annotation_status == TaskStatus.COMPLETED)
                    .filter(
                        ~session.query(LeFormatConvertDB)
                        .filter(LeFormatConvertDB.dataset_uuid == DmvAnnotationDB.dataset_uuid)
                        .exists()
                    )
                    .all()
                )
            else:
                results = (
                    session.query(DmvAnnotationDB)
                    .filter(DmvAnnotationDB.annotation_status == TaskStatus.COMPLETED)
                    .filter(
                        ~session.query(LeFormatConvertDB)
                        .filter(LeFormatConvertDB.dataset_uuid == DmvAnnotationDB.dataset_uuid)
                        .filter(DmvAnnotationDB.device_model == self.specific_device_model)
                        .exists()
                    )
                    .all()
                )
        

        if not results:
            return None

        for item in results:
            dataset_item = (
                session.query(DatasetDB).filter(DatasetDB.dataset_uuid == item.dataset_uuid).first()
            )
            dataset_path = str(Path(dataset_item.yaml_file_path).parent)
            dataset_name = dataset_item.dataset_name
            leformat_path = str(
                Path(self.convert_root_path) / f"{item.device_model}_{dataset_name}"
            )

            leformat_name = f"{item.device_model.lower()}_{dataset_name.lower()}"

            client_log_path = Path(self.convert_root_path) / "client_logs" / leformat_name

            # 在同一个 session 或新开一个
            upsert_leformat_convert(
                session=session,
                ds_uuid=item.dataset_uuid,
                convert_status=TaskStatus.PROCESSING,
            )

            converter_module_path, converter_class_name, converter_config = (
                self._get_converter_module_class_config(
                    device_model=item.device_model, device_model_version=item.device_model_version
                )
            )

            repo_id = f"{ROBOCOIN_PLATFORM}/{leformat_name}"
            return {
                DATASET_UUID: item.dataset_uuid,
                DATASET_NAME: dataset_name,
                LEFORMAT_PATH: leformat_path,
                DATASET_PATH: dataset_path,
                DEVICE_MODEL: item.device_model,
                CONVERTER_CONFIG: converter_config,
                CONVERTER_MODULE_PATH: converter_module_path,
                CONVERTER_CLASS_NAME: converter_class_name,
                VIDEO_BACKEND: self.video_backend,
                IMAGE_WRITER_PROCESSES: self.image_writer_processes,
                IMAGE_WRITER_THREADS: self.image_writer_threads,
                CONVERTER_LOG_DIR: str(client_log_path),
                REPO_ID: repo_id,
                CONVERTER_LOG_NAME: leformat_name,
            }
        return None

    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        ds_uuid = task_content.get(DATASET_UUID)
        leformat_path = task_content.get(LEFORMAT_PATH, "")

        task_status = task_result_content.get(TASK_RESULT_STATUS)
        task_status_msg = task_result_content.get(ERR_MSG)

        convert_status = TaskStatus.COMPLETED if task_status == TASK_SUCCESS else TaskStatus.FAILED

        with self.db.with_session() as session:
            upsert_leformat_convert(
                session=session,
                ds_uuid=ds_uuid,
                convert_status=convert_status,
                leformat_path=leformat_path,
                err_message=task_status_msg,
            )
            self.logger.info(
                f"Upsert {ds_uuid} convert status to {convert_status}, update_message: {task_status_msg}"
            )
