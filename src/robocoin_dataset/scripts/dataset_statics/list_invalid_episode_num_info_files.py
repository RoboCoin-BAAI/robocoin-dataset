from pathlib import Path

import yaml


def list_invalid_episode_num_info_files(
    local_ds_info_hub_file: Path, episode_frames_num_info_files_root_path: Path
) -> None:
    """
    Collect dataset statics from the given dataset info file and directory.

    Args:
        dataset_info_file (str): Path to the dataset info YAML file.
        dataset_dir (str): Path to the dataset directory.

    Returns:
        str: YAML formatted string of the collected dataset statics.
    """

    ds_info_dict: dict[str, str] = {}
    if not local_ds_info_hub_file.exists():
        raise FileNotFoundError(f"Dataset info file {local_ds_info_hub_file} not found.")
    with open(local_ds_info_hub_file) as file:
        ds_info_dict = yaml.safe_load(file)

    root_path = episode_frames_num_info_files_root_path.expanduser().absolute()
    for file in root_path.rglob("*.yaml"):
        print(f"checking {file.name}")
        with open(file) as f:
            data = yaml.safe_load(f)
            episode_frames_num = data["episode_frames_num"]
            if not episode_frames_num or episode_frames_num is None:
                file_path = (local_ds_info_hub_file.parent / file.name).expanduser().absolute()
                file_path_str = str(file_path)
                print(f"invalid episode frame num info file {ds_info_dict[file_path_str]}")


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
        "--root_path",
        type=Path,
        default=Path("outputs/collected_dataset_statics"),
    )

    local_ds_info_hub_file = Path(argparser.parse_args().local_ds_info_hub_file)
    root_path = Path(argparser.parse_args().root_path)

    list_invalid_episode_num_info_files(local_ds_info_hub_file, root_path)


if __name__ == "__main__":
    main()


""" usage:
list_invalid_episode_num_info_files ./outputs/collected_dataset_infos/local_dataset_info_hub.yml --root_path=./outputs/collected_dataset_statics
"""
