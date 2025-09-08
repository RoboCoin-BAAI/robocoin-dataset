import argparse
from pathlib import Path

import yaml


def list_devices(local_dataset_info_hub: Path) -> list[str]:
    file_path = Path(local_dataset_info_hub).expanduser().absolute()
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path) as file_path:
        yaml_dict = yaml.safe_load(file_path)

    local_yaml_files = list(yaml_dict.keys())

    devices = []

    for file_path in local_yaml_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} does not exist.")

        try:
            with open(file_path) as f:
                yaml_dict = yaml.safe_load(f)
                task_devices = yaml_dict["device_model"]
                devices.extend(task_devices)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return list(set(devices))


def main() -> None:
    argparser = argparse.ArgumentParser(description="Collect dataset info files.")

    argparser.add_argument(
        "file_path",
        type=str,
        default="datas",
        help="Path to the local_dataset_info_hub.yml file",
    )

    # 过滤掉可能的空字符串
    file_path = argparser.parse_args().file_path
    for device in list_devices(file_path):
        print(device)


if __name__ == "__main__":
    main()


""" usage:
list_devices ./outputs/collected_dataset_infos/local_dataset_info_hub.yml
"""
