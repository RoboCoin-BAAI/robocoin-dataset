import importlib
import logging
from pathlib import Path

from robocoin_dataset.format_convertors.tolerobot.constant import CLASS_KEY, MODULE_KEY
from robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
)


class LerobotFormatConvertorFactory:
    @staticmethod
    def create_convertor(
        dataset_path: Path,
        device_model: str,
        output_path: Path,
        convertor_config: dict,
        factory_config: dict,
        repo_id: str,
        logger: logging.Logger | None = None,
    ) -> LerobotFormatConvertor:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        if not factory_config:
            raise ValueError("Factory config is empty.")

        if device_model not in factory_config:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        class_config = factory_config[device_model]
        module_path = class_config[MODULE_KEY]
        class_name = class_config[CLASS_KEY]

        module = importlib.import_module(module_path)
        convertor_class = getattr(module, class_name)
        return convertor_class(
            dataset_path=dataset_path,
            output_path=output_path,
            convertor_config=convertor_config,
            repo_id=repo_id,
            logger=logger,
        )
