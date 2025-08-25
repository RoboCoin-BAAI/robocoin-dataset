from pathlib import Path

import yaml
from tqdm import tqdm

from .dataset_statics import DatasetStatics
from .dataset_statics_collector import DatasetStaticsCollector
from .dataset_statics_collector_factory import DatasetStaticsCollectorFactory


def collect_dataset_statics(local_ds_info_hub_file: Path, output_path: Path) -> None:
    """
    Collect dataset statics from the given dataset info file and directory.

    Args:
        dataset_info_file (str): Path to the dataset info YAML file.
        dataset_dir (str): Path to the dataset directory.

    Returns:
        str: YAML formatted string of the collected dataset statics.
    """

    output_path = output_path.expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    ds_info_dict: dict[str, str] = {}
    if not local_ds_info_hub_file.exists():
        raise FileNotFoundError(f"Dataset info file {local_ds_info_hub_file} not found.")
    with open(local_ds_info_hub_file) as file:
        ds_info_dict = yaml.safe_load(file)

    for key, value in tqdm(ds_info_dict.items(), desc="Collecting dataset statics"):
        ds_info_yaml_file = Path(key)
        ds_path = Path(value).parent
        yaml_file_path = output_path / ds_info_yaml_file.name
        if yaml_file_path.exists():
            continue

        if not ds_info_yaml_file.exists():
            continue

        config_file = Path(__file__).parent / "collected_dataset_statics_factory_config.yml"

        try:
            collector: DatasetStaticsCollector = DatasetStaticsCollectorFactory(
                config_file
            ).create_collector(ds_info_yaml_file, ds_path)
            ds_statics: DatasetStatics = collector.collect()
        except Exception:
            # print(f"Failed to collect dataset statics for {ds_path}, {e}")
            continue

        yaml_str = ds_statics._to_yaml()
        try:
            with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
                yaml_file.write(yaml_str)
                print(f"Collected dataset statics saved to {yaml_file_path}")
        except Exception as e:
            print(f"Failed to save dataset statics to {yaml_file_path}: {e}")


def main() -> None:
    import argparse

    argparser = argparse.ArgumentParser(description="Collect dataset statics.")

    argparser.add_argument(
        "local_ds_info_hub_file",
        type=Path,
        default=Path("outputs/collected_dataset_infos/local_dataset_info_hub.yml"),
        help="Path to the local_dataset_info_hub.yml file",
    )
    argparser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/collected_dataset_statics"),
        help="Path to save the collected dataset statics YAML files",
    )
    local_ds_info_hub_file = Path(argparser.parse_args().local_ds_info_hub_file)
    output_path = Path(argparser.parse_args().output_path)
    collect_dataset_statics(local_ds_info_hub_file, output_path)


if __name__ == "__main__":
    main()


""" usage:
collect_dataset_statics ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --output_path=./outputs/collected_dataset_statics/
"""
