import argparse
import re
import uuid
from pathlib import Path

from ruamel.yaml import YAML


def transform_yaml_objects(file_path: Path, output_path: Path) -> None:
    # 创建 YAML 对象，启用注释和顺序保留
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # 防止自动换行

    accepted_object_keys = ["object_name", "level1", "level2", "level3", "level4", "level5"]
    # 读取原始文件
    with open(file_path, encoding="utf-8") as f:
        data = yaml.load(f)

    # 修改 objects 列表
    if "objects" in data and isinstance(data["objects"], list):
        new_objects = []
        for item in data["objects"]:
            if not isinstance(item, dict) or len(item) != 1:
                # 如果格式异常，保留原样
                new_objects.append(item)
                continue

        regularized_objects = []
        for item in new_objects:
            new_object = {}
            if item is None:
                continue
            for key in item:
                if key not in accepted_object_keys:
                    # print(f"objects 中的键 {key} 不被接受")
                    object_name = re.sub(r"\s+", "_", key)
                    new_object["object_name"] = object_name
                    # print(f"objects 中的键 {key} 被替换为 {object_name}")
                else:
                    # print(f"objects 中的键 {key} 被接受: {item[key]}")
                    if isinstance(item[key], str):
                        # print(f"objects 中的键 {key} 的值 {item[key]} 是字符串")
                        new_object[key] = re.sub(r"\s+", "_", item[key])
                        # print(f"objects 中的键 {key} 的值 {item[key]} 被替换为 {new_object[key]}")
                    else:
                        new_object[key] = item[key]

            regularized_objects.append(new_object)

        data["objects"] = regularized_objects

    # 写回文件
    for item in data:
        if isinstance(data[item], str):
            data[item] = re.sub(r"\s+", "_", data[item])
            data[item] = re.sub(r"_+", "_", data[item])
    output = output_path / file_path.name
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    print(f"✅ 已转换: {file_path} -> {output}")


def add_uuid_to_local_dataset_info_file(data: dict) -> dict:
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

        filtered_data = [item for item in objects if isinstance(item, dict)]
        for obj in filtered_data:
            if not isinstance(obj, dict):
                continue

            if not obj:
                continue
            if obj is None:
                continue
            for key in obj:
                if key not in accepted_keys:
                    object_name = re.sub(r"\s+", "_", key)
                    obj.pop(key)
                    obj["object_name"] = object_name
                    break

        data["objects"] = filtered_data

    return data


def fix_dataset_info_files(root_path: Path, output_path: Path) -> None:
    """
    Collect dataset info files from the specified root path and save them to the output path.

    Args:
        root_path (Path): The root directory containing dataset directories.
        output_path (Path): The path where the collected dataset info YAML file will be saved.
    """
    # Ensure the output directory exists

    # Todo: Implement the actual functionality here, lishenyu

    output_path = output_path.expanduser().absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    root_path = root_path.expanduser().absolute()
    if not root_path.exists():
        raise ValueError(f"{root_path} does not exist.")

    ds_info_files = root_path.rglob("local_dataset*")

    counter = 0
    for file in ds_info_files:
        if file.is_file():
            print(f"Processing {file}...")
            transform_yaml_objects(file, output_path=output_path)
            counter += 1
            print(counter)

    # for file in ds_info_files:
    #     if file.is_file():
    #         with file.open("r") as f:
    #             data = yaml.safe_load(f)
    #             try:
    #                 data = add_uuid_to_local_dataset_info_file(data)
    #                 data = fix_yaml_objects_dict(data)
    #                 target_file_path = output_path / file.name
    #                 with target_file_path.open("w") as tf:
    #                     yaml.dump(data, tf)
    #             except Exception as e:
    #                 print(f"Failed to fix dataset info file {file}: {e}")


def main() -> None:
    argparser = argparse.ArgumentParser(description="fix dataset info files.")

    argparser.add_argument(
        "root_path",
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
    root_path_str = argparser.parse_args().root_path
    output_path = Path(argparser.parse_args().output_path)
    root_path = Path(root_path_str)

    fix_dataset_info_files(root_path=root_path, output_path=output_path)


if __name__ == "__main__":
    main()


""" usage:
collect_dataset_info_files "/mnt/nas/1.上传数据/data1, /mnt/nas/1.上传数据/data4, /mnt/nas/1.上传数据/WD14T, /mnt/nas/1.上传数据/新加卷Q, /mnt/nas/七月数据, /mnt/nas/八月数据/星海图,  /mnt/nas/八月数据/松灵分体, /mnt/nas/八月数据/松灵合体, /mnt/nas/八月数据/求之" --output_path=./outputs/collected_dataset_infos/
fix_dataset_info_files outputs/collected_dataset_infos/ --output_path=outputs/fixed_dataset_infos/
"""
