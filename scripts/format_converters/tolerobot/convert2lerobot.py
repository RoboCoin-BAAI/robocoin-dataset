import argparse
import logging
import traceback
from multiprocessing import cpu_count
from pathlib import Path

import yaml
from tqdm import tqdm

from robocoin_dataset.format_converter.tolerobot.constant import (
    DEFAULT_DEVICE_MODEL_VERSION,
    DEVICE_MODEL_VERSION_KEY,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
    LerobotFormatConverterFactory,
)
from robocoin_dataset.utils.logger import setup_logger


def convert2lerobot(
    device_model: str,
    dataset_path: Path,
    output_path: Path,
    factory_config_path: Path,
    repo_id: str,
    log_dir: Path,
    video_backend: str = "pyav",
    image_writer_processes: int = 4,
    image_writer_threads: int = 4,
    converter_log_dir: Path | None = None,
    device_model_version: str | None = None,
) -> LerobotFormatConverter:
    """
    Convert dataset to lerobot format.

    Args:
        dataset_path (str): Path to dataset.
        output_path (str): Path to output directory.
    """

    if not factory_config_path.exists():
        raise FileNotFoundError(f"Factory config file {factory_config_path} does not exist.")

    with open(factory_config_path) as f:
        factory_config = yaml.safe_load(f)
        device_model_configs_list = factory_config[device_model]
        if device_model_configs_list is None:
            raise ValueError(f"Device model {device_model} not found in factory config.")

        if not isinstance(device_model_configs_list, list):
            raise ValueError(f"Device model {device_model} config must be a list.")

        converter_module_path: str | None = None
        for device_model_config in device_model_configs_list:
            if device_model_version is None:
                if device_model_config[DEVICE_MODEL_VERSION_KEY] == DEFAULT_DEVICE_MODEL_VERSION:
                    converter_module_path = device_model_config["module"]
                    converter_class_name = device_model_config["class"]
                    converter_config_path = (
                        factory_config_path.parent / device_model_config["converter_config_path"]
                    )
                break
            if device_model_config[DEVICE_MODEL_VERSION_KEY] == device_model_version:
                converter_module_path = device_model_config["module"]
                converter_class_name = device_model_config["class"]
                converter_config_path = (
                    factory_config_path.parent
                    / factory_config[device_model]["converter_config_path"]
                )
                break
        if converter_module_path is None:
            raise ValueError(
                f"Device model {device_model} with version {device_model_version} not found in factory config."
            )

    with open(converter_config_path) as f:
        converter_config = yaml.safe_load(f)

    logger: logging.Logger = setup_logger(
        name="LEROBOT_CONVERTER",
        log_dir=log_dir,
        level=logging.INFO,
    )

    converter: LerobotFormatConverter = LerobotFormatConverterFactory.create_converter(
        dataset_path=dataset_path,
        device_model=device_model,
        output_path=output_path,
        converter_config=converter_config,
        converter_module_path=converter_module_path,
        converter_class_name=converter_class_name,
        repo_id=repo_id,
        video_backend=video_backend,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        logger=logger,
    )

    total_episodes = converter.get_episodes_num()

    for task, task_ep_idx, ep_idx in tqdm(
        converter.convert(),
        total=total_episodes,
        desc="Converting Dataset",
        unit="episode",
    ):
        logger.info(f"Converted episode {task_ep_idx} of task {task}, total ep_idx is:{ep_idx}")


def main() -> None:
    argparser = argparse.ArgumentParser(description="Convert dataset to lerobot format.")
    # ... 保留原有参数设置 ...
    argparser.add_argument(
        "--num_processes",
        type=int,
        default=int(cpu_count() / 4),  # 默认情况下，使用所有可用的CPU核心/4
        help="Number of processes to use for the conversion (default: all available cores)",
    )
    argparser.add_argument(
        "--dataset_path",
        type=Path,
    )
    argparser.add_argument(
        "--output_path",
        type=Path,
    )
    argparser.add_argument(
        "--device_model",
        type=str,
    )
    argparser.add_argument(
        "--factory_config_path",
        type=Path,
        default=Path("configs/converters/lerobot_format_convertor_factory_config.yaml"),
    )
    argparser.add_argument(
        "--repo_id",
        type=str,
        default="",
    )
    argparser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("outputs/lerobot_converter/logs/"),
    )
    argparser.add_argument(
        "--image_writer_processes",
        type=int,
        default=4,
        help="Number of processes for image writing (default: 4)",
    )
    argparser.add_argument(
        "--image_writer_threads",
        type=int,
        default=4,
        help="Number of threads per process for image writing (default: 2)",
    )
    argparser.add_argument(
        "--video_backend",
        type=str,
        choices=["pyav", "opencv"],
        default="pyav",
        help="Backend for video processing (default: pyav)",
    )

    argparser.add_argument(
        "--device_model_version",
        type=str,
        default="none",
        help="Device model version (default: none)",
    )

    args = argparser.parse_args()
    device_model_version = None
    if args.device_model_version != "none":
        device_model_version = args.device_model_version

    try:
        convert2lerobot(
            device_model=args.device_model,
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            factory_config_path=args.factory_config_path,
            repo_id=args.repo_id,
            log_dir=args.log_dir,
            video_backend=args.video_backend,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            converter_log_dir=args.log_dir,
            device_model_version=device_model_version,
        )
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()

"""_summary_

python scripts/format_converters/tolerobot/convert2lerobot.py \
--dataset_path /home/lxc/Downloads/stir_coffee/stir_coffee_1 \
--output_path ./outputs/lerobot_converter_test/stir_coffee_1 \
--device_model realman_rmc_aidal \
--factory_config_path scripts/format_converters/tolerobot/configs/converter_factory_config_test.yaml \
--repo_id robocoin/realman_rmc_aidal_stir_coffee \
--log_dir ./outputs/robocoin/logs \
--image_writer_processes 10 \
--image_writer_threads 4 \
--video_backend pyav

python scripts/format_converters/tolerobot/convert2lerobot.py \
--dataset_path /mnt/nas/11realman_rmc_aidal/pour_bowl_repeatedly \
--output_path ./outputs/lerobot_converter/pour_bowl_repeatedly \
--device_model realman_rmc_aidal \
--factory_config_path scripts/format_converters/tolerobot/configs/converter_factory_config_test.yaml \
--repo_id robocoin/realman_rmc_aidal_stir_coffee \
--log_dir ./outputs/robocoin/logs \
--image_writer_processes 10 \
--image_writer_threads 4 \
--video_backend pyav

"""
