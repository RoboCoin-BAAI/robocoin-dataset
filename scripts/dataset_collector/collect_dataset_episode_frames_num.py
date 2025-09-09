import logging
from datetime import datetime
from pathlib import Path

import yaml
from tqdm.contrib.concurrent import thread_map

from robocoin_dataset.dataset_statics.episode_frames_collector import EpisodeFramesCollector
from robocoin_dataset.dataset_statics.episode_frames_collector_factory import (
    EpisodeFramesCollectorFactory,
)


def collect_dataset_episode_frame_num(
    local_ds_info_hub_file: Path,
    output_path: Path,
    device_model: str = "",
    concurrent_workers: int = 1,
) -> None:
    """
    Collect dataset statics from the given dataset info file and directory.

    Args:
        dataset_info_file (str): Path to the dataset info YAML file.
        dataset_dir (str): Path to the dataset directory.

    Returns:
        str: YAML formatted string of the collected dataset statics.
    """
    logger = logging.getLogger(name="COLLECT_DS_EPISODE_FRAMES_NUM")
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        (output_path.expanduser().absolute() / "logs").mkdir(parents=True, exist_ok=True)
        logger_path = (
            output_path.expanduser().absolute() / "logs" / f"collect_dataset_{current_time}.log"
        )
        file_handler = logging.FileHandler(logger_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    output_path = output_path.expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    ds_info_dict: dict[str, str] = {}
    if not local_ds_info_hub_file.exists():
        raise FileNotFoundError(f"Dataset info file {local_ds_info_hub_file} not found.")
    with open(local_ds_info_hub_file) as file:
        ds_info_dict = yaml.safe_load(file)

    def process_item(key_value_tuple: tuple[str, str]) -> None:
        ds_info_yaml_file = Path(key_value_tuple[0])
        ds_path = Path(key_value_tuple[1]).parent
        episode_frames_num: list[int] = []
        if not ds_info_yaml_file.exists():
            logger.error(f"Dataset info file {ds_info_yaml_file} not found.")
            return

        with open(ds_info_yaml_file) as file:
            ds_info_yaml = yaml.safe_load(file)
            if not ds_info_yaml:
                logger.error(f"Dataset info file {ds_info_yaml_file} is empty.")
                return

            ds_device_models = ds_info_yaml.get("device_model", [])
            ds_device_model = ds_device_models[0] if ds_device_models else ""

        if device_model and (ds_device_model != device_model):
            return

        yaml_file_path = output_path / ds_device_model / ds_info_yaml_file.name
        if yaml_file_path.exists():
            logger.warning(f"Skipping dataset {yaml_file_path} because it already exists")
            return

        try:
            collector: EpisodeFramesCollector = EpisodeFramesCollectorFactory().create_collector(
                ds_info_yaml_file, ds_path
            )
            logger.info(f"collecting dataset episode frames num for {ds_path} ...")
            episode_frames_num = collector.collect()
        except Exception as e:
            logger.error(f"Failed to collect dataset episode frames num for {ds_path}: {e}")
            return

        ds_info_yaml["episode_frames_num"] = episode_frames_num
        yaml_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(ds_info_yaml, yaml_file, default_flow_style=False)
            logger.info(f"Dataset episode frames saved to {yaml_file_path}")
            return
        except Exception as e:
            logger.error(f"Failed to save dataset episode frames num to {yaml_file_path}: {e}")

    thread_map(
        process_item,
        ds_info_dict.items(),
        max_workers=concurrent_workers,
        desc="Collecting dataset episode frames num",
        unit="Item",
    )


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser(description="Collect dataset statics.")

    argparser.add_argument(
        "local_ds_info_hub_file",
        type=Path,
        default=Path("outputs/collected_dataset_infos/local_dataset_info_hub.yml"),
        help="Path to the local_dataset_info_hub.yml file",
    )


if __name__ == "__main__":
    main()


""" usage:

collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics --concurrent_workers=30

# hdf5 format: 
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=agilex_cobot_magic_aloha --concurrent_workers=30
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=agilex_cobot_decoupled_magic --concurrent_workers=30
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=realman_rmc_aidal --concurrent_workers=1

# pika format:
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=agilex_pika_sense --concurrent_workers=1
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=agilex_pika_sense_single --concurrent_workers=1

# galaxea_r1_lite format:
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=galaxea_r1_lite --concurrent_workers=1

# lerobot format:
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=adora_dual_v1 --concurrent_workers=1

# unitree format:
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=unitree_g1 --concurrent_workers=1

# vision_pro format:
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=human --concurrent_workers=1
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=visionpro --concurrent_workers=1

# mmk format:
# OK
collect_dataset_episode_frames_num ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/ --device_model=discover_robotics_aitbot_mmk2 --concurrent_workers=1
"""
