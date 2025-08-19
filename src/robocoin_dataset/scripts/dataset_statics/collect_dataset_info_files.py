import argparse
import os
from pathlib import Path
import shutil
import yaml


def collect_dataset_info_files(root_path: Path, output_path: Path) -> None:
    """
    Collect dataset info files from the specified root path and save them to the output path.

    Args:
        root_path (Path): The root directory containing dataset directories.
        output_path (Path): The path where the collected dataset info YAML file will be saved.
    """
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Todo: Implement the actual functionality here, lishenyu
    
    hub_mapping = {}
    counter = 0
    for current_path, subdirs, files in os.walk(root_path):
        if "local_dataset_info.yml" in files:
            counter += 1
            src = Path(current_path) / "local_dataset_info.yml"
            dst = output_path / f"locao_dataset_info_{counter}.yml"
            if not src.exists():
                continue
            shutil.copy2(src, dst)
            hub_mapping[dst.name] = str(Path(current_path).relative_to(root_path))
            subdirs.clear()        

    (output_path / "local_dataset_info_hub.yml").write_text(
        yaml.dump(hub_mapping, sort_keys=True, allow_unicode=True),
        encoding="utf-8"
    )



    print(f"Collecting dataset info files from {root_path} and saving to {output_path}")




def main() -> None:
    argparser = argparse.ArgumentParser(description="Collect dataset info files.")

    argparser.add_argument(
        "root_path",
        type=Path,
        default=Path("datas"),
        help="Path to the root directory containing the dataset directories",
    )
    argparser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/collected_dataset_infos"),
        help="Path to save the collected dataset info YAML file",
    )
    root_path = Path(argparser.parse_args().root_path)
    output_path = Path(argparser.parse_args().output)
    collect_dataset_info_files(root_path, output_path)


if __name__ == "__main__":
    main()


""" usage:
collect_dataset_info_files /mnt/nas/datas/ ./outputs/collected_dataset_infos/

函数作用：
1. 输入参数：root_path, output_path
2. 函数功能：
 (1) 如果output_path目录不存在，则创建该目录
 (2) 收集root_path目录下的所有子目录的local_dataset_info.yml文件，重命名为locao_dataset_info_1.yml并保存到output_path目录下
 (3) 另外新建一个local_dataset_info_hub.yml文件, 内容示例如下：
```yaml
locao_dataset_info_1: /mnt/nas/datas/pika/pickup_apples/
```

"""
