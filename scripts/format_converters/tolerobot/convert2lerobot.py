import argparse
import logging
import sys
from datetime import datetime
from multiprocessing import Process, cpu_count
from pathlib import Path

import yaml
from tqdm import tqdm

from robocoin_dataset.format_converter.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
    LerobotFormatConvertorFactory,
)


def setup_logger(
    log_name: str,
    log_path: str | Path,
    level=logging.INFO,  # noqa: ANN001
) -> logging.Logger:
    """
    创建一个同时输出到文件（带时间戳）和控制台的日志记录器

    Args:
        name: logger 名称（建议用 __name__）
        log_dir: 日志存储目录路径
        log_name: 日志文件名前缀
        level: 日志级别

    Returns:
        配置好的 logger
    """
    # 1. 创建 logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复输出

    # 2. 创建日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{log_name}_{timestamp}.log"
    log_path = Path(log_path) / log_filename

    # 4. 确保日志目录存在
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 文件处理器
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # 6. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 7. 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # 可选：记录日志文件路径
    logger.info(f"日志已启用，输出到文件: {log_path.resolve()}")

    return logger


def convert2lerobot(
    device_model: str,
    dataset_path: Path,
    output_path: Path,
    factory_config_path: Path,
    repo_id: str,
    log_path: Path,
    **kwargs: dict,
) -> LerobotFormatConvertor:
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

    if device_model not in factory_config:
        raise ValueError(f"Device model {device_model} not found in factory config.")

    convertor_config_path = (
        factory_config_path.parent / factory_config[device_model]["convertor_config_path"]
    )

    with open(convertor_config_path) as f:
        convertor_config = yaml.safe_load(f)

    logger = setup_logger(log_name="LEROBOT_CONVERTOR", level=logging.INFO, log_path=log_path)

    convertor = LerobotFormatConvertorFactory.create_convertor(
        dataset_path=dataset_path,
        device_model=device_model,
        factory_config=factory_config,
        convertor_config=convertor_config,
        repo_id=repo_id,
        logger=logger,
        output_path=output_path,
        **kwargs,
    )

    total_episodes = convertor.get_episodes_num()
    for task, task_ep_idx, ep_idx in tqdm(
        convertor.convert(),
        total=total_episodes,
        desc="Converting Dataset",
        unit="episode",
    ):
        logger.info(f"Converted episode {task_ep_idx} of task {task}, total ep_idx is:{ep_idx}")


def worker(args: dict, output_suffix: int) -> None:
    # 创建新的输出路径
    args.output_path = args.output_path / f"output_{output_suffix}"
    # 执行转换操作
    args_dict = vars(args)
    args_dict.pop("num_processes", None)
    convert2lerobot(**vars(args))


def main_mp() -> None:
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
        default=Path("configs/convertors/lerobot_format_convertor_factory_config.yaml"),
    )
    argparser.add_argument(
        "--repo_id",
        type=str,
        default="",
    )
    argparser.add_argument(
        "--log_path",
        type=Path,
        default=Path("outputs/lerobot_convertor/logs/"),
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
    args = argparser.parse_args()

    processes = []
    for i in range(args.num_processes):
        # 对于每个进程，我们传递相同的参数，但指定不同的输出后缀
        p = Process(target=worker, args=(args, i + 1))
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()


if __name__ == "__main__":
    main_mp()

"""_summary_

python scripts/format_converters/tolerobot/convert2lerobot.py \
--dataset_path /home/lxc/Downloads/stir_coffee/stir_coffee_2 \
--output_path ./outputs/lerobot_converter_test/ \
--device_model realman_rmc_aidal \
--factory_config_path scripts/format_converters/tolerobot/configs/convertor_factory_config.yaml \
--repo_id robocoin/realman_rmc_aidal_stir_coffee \
--log_path ./outputs/robocoin/logs \
--image_writer_processes 4 \
--image_writer_threads 4 \
--video_backend opencv \
--num_processes 8

"""
