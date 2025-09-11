import io
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from PIL import Image

from robocoin_dataset.format_converter.tolerobot.constant import (
    ARGS_KEY,
    DESCRIPTION_TXT,
    FEATURES_KEY,
    H5_SUFFIX,
    HDF5_SUFFIX,
    LOCAL_DATASET_INFO_FILE,
    LOCAL_TASK_INFO_FILE,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)


@dataclass
class H5Buffer:
    h5_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


ALLOWED_RULES = {
    "exact_names": {
        LOCAL_TASK_INFO_FILE,
        LOCAL_DATASET_INFO_FILE,
        DESCRIPTION_TXT,
    },  # 允许的完整文件名
    "suffixes": {H5_SUFFIX, HDF5_SUFFIX},  # 允许的后缀
}


def is_allowed_file(file_path: Path) -> bool:
    filename = file_path.name
    suffix = file_path.suffix

    return filename in ALLOWED_RULES["exact_names"] or suffix in ALLOWED_RULES["suffixes"]


def find_unexpected_files(directory: Path, include_hidden: bool = False) -> list[str]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"路径不是目录: {directory}")

    unexpected_files: list[str] = []

    # 🔍 递归遍历所有文件
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            # 跳过隐藏文件（可选）
            if not include_hidden and file_path.name.startswith("."):
                continue

            if not is_allowed_file(file_path):
                unexpected_files.append(str(file_path))

    return unexpected_files


class LerobotFormatConverterHdf5(LerobotFormatConverter):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        self.h5_buffer: H5Buffer = H5Buffer()
        self._image_is_iobytes = True

        super().__init__(
            dataset_path=dataset_path,
            output_path=output_path,
            converter_config=converter_config,
            repo_id=repo_id,
            device_model=device_model,
            logger=logger,
            video_backend=video_backend,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )

    def _prevalidate_files(self) -> None:
        unexpected_files: list[Path] = []
        for path in self.path_task_dict.keys():
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist")
            if path.is_file():
                raise ValueError(f"{path} is a file")

            unexpected_files.extend(find_unexpected_files(path))

        if unexpected_files:
            err_msg = "Found unexpected files in dataset directory:"
            err_msg += "\n".join(f"{path}" for path in unexpected_files)
            raise Exception(err_msg)

    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        if not images_buffer:
            images_buffer = self._prepare_episode_images_buffer(task_path, ep_idx)

        h5_path = args_dict["h5_path"]
        image_data = images_buffer[h5_path][frame_idx]

        if self._image_is_iobytes:
            try:
                img = Image.open(io.BytesIO(image_data))
                return np.array(img)
            except Exception:
                self._image_is_iobytes = False
        return image_data

    # @override
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        h5_path = args_dict["h5_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        return sub_states_buffer[h5_path][frame_idx][from_idx:to_idx]

    # @override
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        h5_path = args_dict["h5_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        return sub_actions_buffer[h5_path][frame_idx][from_idx:to_idx]

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        args = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SUB_STATE_KEY][0][
            ARGS_KEY
        ]
        if "h5_path" not in args:
            raise ValueError("h5_path is not specified in the config")
        h5_path = args["h5_path"]

        h5_file_path = self.task_episode_h5file_paths[task_path][ep_idx]
        try:
            with h5py.File(h5_file_path, "r") as h5_file:
                return h5_file[h5_path].shape[0]
        except Exception as e:
            raise ValueError(f"Error while reading h5 file {h5_file_path}: {e}")

    # @override
    def _get_task_episodes_num(self, task_path: Path) -> int:
        return len(self.task_episode_h5file_paths[task_path])

    # @override
    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    # @override
    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    # @override
    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    @cached_property
    def task_episode_h5file_paths(self) -> dict[Path, list[Path]]:
        task_episode_paths = {}
        for path in self.path_task_dict.keys():
            if path.exists():
                h5_files = natsorted(list(path.rglob("*.h5")))
                h5_files.extend(natsorted(list(path.rglob("*.hdf5"))))
                task_episode_paths[path] = h5_files
        return task_episode_paths

    def _get_episode_h5_data(self, task_path: Path, ep_idx: int) -> any:
        should_load = self.h5_buffer.task_path != task_path or self.h5_buffer.ep_idx != ep_idx
        if not should_load:
            return self.h5_buffer.h5_data
        self.h5_buffer.h5_data = {}

        def _get_dataset(name: str, obj: any) -> None:
            if isinstance(obj, h5py.Dataset):
                self.h5_buffer.h5_data[name] = obj[()]

        with h5py.File(self.task_episode_h5file_paths[task_path][ep_idx]) as h5_file:
            h5_file.visititems(_get_dataset)
            self.h5_buffer.task_path = task_path
            self.h5_buffer.ep_idx = ep_idx

        return self.h5_buffer.h5_data
