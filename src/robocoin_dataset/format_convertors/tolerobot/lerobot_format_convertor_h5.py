import io
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from PIL import Image

from robocoin_dataset.format_convertors.tolerobot.constant import (
    ARGS_KEY,
    FEATURES_KEY,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
)


@dataclass
class H5Buffer:
    h5_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


class LerobotFormatConvertorHdf5(LerobotFormatConvertor):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        convertor_config: dict,
        repo_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.h5_buffer: H5Buffer = H5Buffer()
        super().__init__(dataset_path, output_path, convertor_config, repo_id, logger)

    # @override
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
        print("h5_override _get_frame_image")
        print(f"h5_path: {h5_path}, keys: {images_buffer.keys()}")
        image_bytes = images_buffer[h5_path][frame_idx]
        img = Image.open(io.BytesIO(image_bytes))
        return np.array(img)

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
        names_num: int,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        h5_path = args_dict["h5_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        return sub_actions_buffer[h5_path][frame_idx][from_idx:to_idx]

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        args = self.convertor_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SUB_STATE_KEY][0][
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
    def _prepare_episode_state_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_h5_data(task_path, ep_idx)

    # @override
    def _prepare_episode_action_buffer(self, task_path: Path, ep_idx: int) -> any:
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
            print(f"buffer is ok, skipping Loading episode {ep_idx} from {task_path}")
            return self.h5_buffer.h5_data
        self.h5_buffer.h5_data = {}

        def _get_dataset(name: str, obj: any) -> np.ndarray:
            if isinstance(obj, h5py.Dataset):
                self.h5_buffer.h5_data[name] = obj[()]

        with h5py.File(self.task_episode_h5file_paths[task_path][ep_idx]) as h5_file:
            h5_file.visititems(_get_dataset)
            self.h5_buffer.task_path = task_path
            self.h5_buffer.ep_idx = ep_idx

        print("h5_file loaded")
        print(self.h5_buffer.h5_data.keys())
        return self.h5_buffer.h5_data
