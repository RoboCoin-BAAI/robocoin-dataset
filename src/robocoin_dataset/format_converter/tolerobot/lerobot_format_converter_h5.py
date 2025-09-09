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
    FEATURES_KEY,
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
        
        # Auto-detect cameras before calling super().__init__
        self._auto_detect_and_update_camera_config(converter_config, Path(dataset_path), logger)
        
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

    def _auto_detect_and_update_camera_config(self, converter_config: dict, dataset_path: Path, logger: logging.Logger | None = None) -> None:
        """
        Auto-detect available cameras in HDF5 files and update the converter config accordingly.
        """
        # Find the first HDF5 file to inspect
        h5_files = list(dataset_path.rglob("*.h5")) + list(dataset_path.rglob("*.hdf5"))
        if not h5_files:
            if logger:
                logger.warning("No HDF5 files found for camera auto-detection")
            return
        
        sample_h5_file = h5_files[0]
        if logger:
            logger.info(f"Auto-detecting cameras from sample file: {sample_h5_file}")
        
        try:
            with h5py.File(sample_h5_file, 'r') as h5_file:
                # Check if observations/images exists
                if 'observations' not in h5_file or 'images' not in h5_file['observations']:
                    if logger:
                        logger.warning("No observations/images group found in HDF5 file")
                    return
                
                # Get available camera names
                available_cameras = list(h5_file['observations/images'].keys())
                if logger:
                    logger.info(f"Detected cameras: {available_cameras}")
                
                # Update the config to only include cameras that actually exist
                if FEATURES_KEY in converter_config and OBSERVATION_KEY in converter_config[FEATURES_KEY]:
                    if 'images' in converter_config[FEATURES_KEY][OBSERVATION_KEY]:
                        original_cameras = converter_config[FEATURES_KEY][OBSERVATION_KEY]['images']
                        updated_cameras = []
                        
                        for camera_config in original_cameras:
                            if ARGS_KEY in camera_config and 'h5_path' in camera_config[ARGS_KEY]:
                                h5_path = camera_config[ARGS_KEY]['h5_path']
                                # Extract camera name from h5_path (e.g., "observations/images/cam_high" -> "cam_high")
                                camera_name = h5_path.split('/')[-1]
                                
                                if camera_name in available_cameras:
                                    updated_cameras.append(camera_config)
                                    if logger:
                                        logger.info(f"Keeping camera config: {camera_config.get('cam_name', camera_name)}")
                                else:
                                    if logger:
                                        logger.warning(f"Removing camera config for non-existent camera: {camera_config.get('cam_name', camera_name)} (path: {h5_path})")
                        
                        # Update the config
                        converter_config[FEATURES_KEY][OBSERVATION_KEY]['images'] = updated_cameras
                        
                        if logger:
                            logger.info(f"Updated camera config to include {len(updated_cameras)} cameras")
                
        except Exception as e:
            if logger:
                logger.error(f"Error during camera auto-detection: {e}")
            # Don't raise the error, just continue with original config
