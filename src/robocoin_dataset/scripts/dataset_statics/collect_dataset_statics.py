from pathlib import Path

import yaml

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

    for key, value in ds_info_dict.items():
        ds_info_yaml_file = Path(key)
        ds_path = Path(value)
        if not ds_info_yaml_file.exists():
            raise FileNotFoundError(f"Dataset info file {ds_info_yaml_file} not found.")
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset directory {ds_path} not found.")

        collector: DatasetStaticsCollector = DatasetStaticsCollectorFactory(
            local_ds_info_hub_file
        ).create_collector(ds_info_yaml_file, ds_path)

        ds_statics: DatasetStatics = collector.collect()
        yaml_str = ds_statics._to_yaml()
        yaml_file_path = output_path / ds_info_yaml_file.name
        try:
            with open(yaml_file_path, "w") as yaml_file:
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
        "--output",
        type=Path,
        default=Path("outputs/collected_dataset_statics"),
        help="Path to save the collected dataset statics YAML files",
    )
    local_ds_info_hub_file = Path(argparser.parse_args().local_ds_info_hub_file)
    output_path = Path(argparser.parse_args().output)
    collect_dataset_statics(local_ds_info_hub_file, output_path)


if __name__ == "__main__":
    main()


""" usage:
collect_dataset_statics ./outputs/collected_dataset_infos ./outputs/collected_dataset_statics/

函数作用：
1. 输入参数：root_path, output_path
2. 函数功能：
 (1) 如果output_path目录不存在，则创建该目录
 (2) 收集root_path目录下的所有子目录的local_dataset_info.yml文件，重命名为locao_dataset_info_1..n.yml并保存到output_path目录下
 (3) 另外新建一个local_dataset_info_hub.yml文件, 内容示例如下：
```yaml
full_path(locao_dataset_info_1): /mnt/nas/datas/pika/pickup_apples/
```
"""
