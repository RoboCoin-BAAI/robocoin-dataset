import argparse
import os
import re
import shutil
import uuid
from pathlib import Path

import yaml


def add_uuid_to_local_dataset_info_file(file: Path) -> dict:
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found.")
    with open(file) as f:
        data = yaml.safe_load(f)
        if "dataset_uuid" not in data:
            ds_uuid_str = str(uuid.uuid4())
            data["dataset_uuid"] = ds_uuid_str

    return data


def fix_yaml_objects_dict(data: dict) -> dict:
    """修复单个 YAML 文件的 objects 结构"""
    accepted_keys = ["object_name", "level1", "level2", "level3", "level4", "level5"]
    if not isinstance(data, dict):
        raise TypeError("输入数据必须是字典")

    if "objects" in data:
        objects = data["objects"]
        if not isinstance(objects, list):
            raise TypeError("objects 必须是一个列表")

        for obj in objects:
            if not isinstance(obj, dict):
                raise TypeError("objects 中的元素必须是字典")

            for key in obj:
                if key not in accepted_keys:
                    data.pop(key)
                    data["object_name"] = key
                    break

    return data


def collect_dataset_info_files(root_paths: list[Path], output_path: Path) -> None:
    """
    Collect dataset info files from the specified root path and save them to the output path.

    Args:
        root_path (Path): The root directory containing dataset directories.
        output_path (Path): The path where the collected dataset info YAML file will be saved.
    """
    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Todo: Implement the actual functionality here, lishenyu

    dirs_to_scan = []

    hub_mapping = {}
    counter = 0
    for root_path in root_paths:
        if not os.path.exists(root_path):
            print(f"{root_path} does not exist.")
            continue

        dirs_to_scan.append(root_path)

    while len(dirs_to_scan) > 0:
        current_path = dirs_to_scan.pop(0)
        files = Path(current_path).glob("local_dataset*")
        if files:
            file_exists = False
            for file in files:
                file_exists = True
                with open(file) as f:
                    pass
                copy_dst = output_path / f"local_dataset_info_{counter}.yml"
                copy_dst_full_path = copy_dst.expanduser().absolute()
                shutil.copy2(file, copy_dst)
                hub_mapping[str(copy_dst_full_path)] = str(file)
                counter += 1
                break
            if file_exists:
                continue

        subdirectories = [item for item in Path(current_path).glob("*") if item.is_dir()]
        dirs_to_scan.extend(subdirectories)

    (output_path / "local_dataset_info_hub.yml").write_text(
        yaml.dump(hub_mapping, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )


def main() -> None:
    argparser = argparse.ArgumentParser(description="Collect dataset info files.")

    argparser.add_argument(
        "root_paths",
        type=str,
        default="datas",
        help="Path to the root directory containing the dataset directories",
    )
    argparser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/collected_dataset_infos"),
        help="Path to save the collected dataset info YAML file",
    )

    # 过滤掉可能的空字符串
    root_paths_str = argparser.parse_args().root_paths
    paths = re.split(r"[\s;,:\n]+", root_paths_str)
    paths = [p for p in paths if p.strip()]
    paths = list(set(paths))
    output_path = Path(argparser.parse_args().output_path)

    collect_dataset_info_files(paths, output_path)


if __name__ == "__main__":
    main()


""" usage:
collect_dataset_info_files "/mnt/nas/1.上传数据/data1, /mnt/nas/1.上传数据/data4, /mnt/nas/1.上传数据/WD14T, /mnt/nas/1.上传数据/新加卷Q, /mnt/nas/七月数据, /mnt/nas/八月数据/星海图,  /mnt/nas/八月数据/松灵分体, /mnt/nas/八月数据/松灵合体, /mnt/nas/八月数据/求之" --output_path=./outputs/collected_dataset_infos/
函数作用：
1. 输入参数：root_path, output_path
2. 函数功能：
 (1) 如果output_path目录不存在，则创建该目录
 (2) 收集root_path目录下的所有子目录的local_dataset_info.yml文件，重命名为locao_dataset_info_1.yml并保存到output_path目录下
 (3) 另外新建一个local_dataset_info_hub.yml文件, 内容示例如下：
```yaml
locao_dataset_info_1: /mnt/nas/datas/pika/pickup_apples/
```
source .venv/bin/activate


"""
