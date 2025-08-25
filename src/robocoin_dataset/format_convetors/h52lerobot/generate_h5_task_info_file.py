import argparse
import uuid
from pathlib import Path

import h5py
import yaml


def generate_h5_task_info_file(h5dir_path: Path) -> None:
    """
    Generate a UUID mark file in the specified H5 directory.

    Arg:
      h5dir_path (str): Path to the H5 directory where the UUID file will be created.
    """

    dataset_paths: list[str] = []

    def _visitor(name: str, obj: any) -> None:
        if isinstance(obj, h5py.Dataset):
            dataset_paths.append(name)  # name 是字符串路径，如 'group/dataset'

    # Ensure the directory exists
    if not h5dir_path:
        raise ValueError("h5dir_path must be provided.")
    if not h5dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {h5dir_path}")

    has_f5_file = False
    items = h5dir_path.iterdir()
    h5_file_dataset_paths: list[str] = []
    for item in items:
        if item.is_file() and item.name.endswith(".hdf5"):
            has_f5_file = True
            with h5py.File(item, "r") as f:
                h5_file_dataset_paths = f.visititems(_visitor)  # 遍历所有项，传入 name 和 obj
            break

    if not has_f5_file:
        raise FileNotFoundError(f"No .hdf5 files found in directory: {h5dir_path}")

    task_info_file = h5dir_path / "h5_task_info.yml"
    if task_info_file.exists():
        raise FileExistsError(f"h5 taks file already exists: {task_info_file}")

    # Generate a UUID
    uuid_value = str(uuid.uuid4())
    yaml_data = {}
    yaml_data["task_uuid"] = uuid_value
    yaml_data["dataset_paths"] = h5_file_dataset_paths

    # Write to YAML file
    with open(task_info_file, "w") as f:
        yaml.dump(yaml_data, f)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        type=str,
        required=True,
        help="Path to the H5 directory where the h5 task info file will be created.",
    )
    h5dir_path = Path(argparser.parse_args())
    generate_h5_task_info_file(h5dir_path)


if __name__ == "__main__":
    main()
