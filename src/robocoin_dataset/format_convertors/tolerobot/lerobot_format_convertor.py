from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml
from deepmerge import Merger

from robocoin_dataset.constant import (
    DEVICE_LIST_KEY,
    DEVICE_TO_FEATURES_FILE,
    LEROBOT_ACTION_KEY,
    LEROBOT_CAM_NAME_KEY,
    LEROBOT_DEFAULT_IMAGE_SHAPE_NAMES,
    LEROBOT_FEATURES_KEY,
    LEROBOT_IMAGE_KEY,
    LEROBOT_OSERVATION_KEY,
    LEROBOT_STATE_KEY,
    LOCAL_DATASET_INFO_FILE,
)


@dataclass
class LerobotFormatConvertorConfig:
    local_dataset_path: str = field(metadata={"help": "Path to the dataset directory."})


class LerobotFormatConvertor:
    """
    Base class for converting datasets to the LeRobot format.
    This class should be extended by specific dataset format converters.
    """

    def __init__(self, dataset_path: str, lerobot_dst_path: str) -> None:
        if not dataset_path:
            raise ValueError("Dataset path must be provided.")
        if not lerobot_dst_path:
            raise ValueError("LeRobot destination path must be provided.")

        if dataset_path == lerobot_dst_path:
            raise ValueError("Dataset path and LeRobot destination path cannot be the same.")
        self.dataset_path = Path(dataset_path).expanduser().absolute()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        self.lerobot_dst_path = Path(lerobot_dst_path).expanduser().absolute()

        self.features = []

    def _get_episode_buffer_data(self, episode_id: int) -> any:
        return None

    def _get_frame_buffer_data(self, episode_id: int, frame_id: int) -> any:
        return None
    @abstractmethod
    def _get_image_shape(self, image_name: str) -> tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def _get_image_channels(self, image_name: str) -> tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def _get_episode_num(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_episode_frame_num(self, episode_id: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_episode_frame_image(
        self, episode_id: int, frame_id:int, image_config: dict, episode_buffers: any, frame_buffer: any
    ) -> np.ndarray:
        raise NotImplementedError

    def _add_episode_frame(
        self, episode_id: int, frame_id: int, episode_buffer: any=None, frame_buffer: any=None
    ) -> None:
        try:
            iamge_configs = self.features[LEROBOT_OSERVATION_KEY][LEROBOT_IMAGE_KEY]
            for image_config in iamge_configs:
                cam_name = image_config[LEROBOT_CAM_NAME_KEY]
                config = self._get_episode_frame_image(image_config, episode_buffer)

        except Exception as e:
            raise ValueError(f"Error add frame {frame_id} from episode {episode_id}: {e}")

    def _get_local_dataset_info(self) -> dict:
        local_dataset_info_file_path = self.dataset_path / LOCAL_DATASET_INFO_FILE
        if not local_dataset_info_file_path.exists():
            raise FileNotFoundError(
                f"Local dataset info file {local_dataset_info_file_path} does not exist."
            )
        self.local_dataset_info_file_path = local_dataset_info_file_path
        return yaml.safe_load(local_dataset_info_file_path)

    def _get_lerobot_features(self, local_dataset_info: dict) -> dict:
        device2features_yaml_file_path = Path(__file__).parent / DEVICE_TO_FEATURES_FILE
        device2features_dir: Path = device2features_yaml_file_path.parent
        if not device2features_yaml_file_path.exists():
            raise FileNotFoundError(
                f"Device to features file {device2features_yaml_file_path} does not exist."
            )
        device2features_dict = yaml.safe_load(device2features_yaml_file_path)

        if DEVICE_LIST_KEY not in local_dataset_info:
            raise ValueError(
                f"Device list key {DEVICE_LIST_KEY} NOT found in {self.local_dataset_info_file_path}."
            )

        device_features: list[dict] = []
        for device in local_dataset_info[DEVICE_LIST_KEY]:
            if device not in device2features_dict:
                raise ValueError(
                    f"Device {device} is not supported. Please check the device list in {device2features_yaml_file_path}."
                )
            device_feature_file: Path = device2features_dir / device2features_dict[device]
            if not device_feature_file.exists():
                raise ValueError(
                    f"Feature file {device_feature_file} does not exist. Please check the device list in {device2features_yaml_file_path}."
                )
            data = yaml.safe_load(device_feature_file)
            if LEROBOT_FEATURES_KEY not in data:
                raise ValueError(
                    f"Feature file {device_feature_file} does not contain 'features' key. Please check the device list in {device2features_yaml_file_path}."
                )

            device_features.append(data[LEROBOT_FEATURES_KEY])

        return LerobotFormatConvertor._merge_and_validate_features(device_features)

    def _collect_features(self, feature_yaml_files: list[Path]) -> None:
        """
        Collect features from the dataset.
        This method should be implemented by subclasses.
        """

        try:
            for feature_yaml_file in feature_yaml_files:
                with open(feature_yaml_file) as f:
                    feature_yaml = yaml.safe_load(f)
                    feature_dict = feature_yaml.get("features", {})
                    self.features.append(feature_dict)
        except Exception as e:
            raise RuntimeError(f"Error loading feature from {feature_yaml_file}: {e}")

    @staticmethod
    def _find_lists_in_dict(input_dict: dict) -> list[list]:
        out_list: list[list] = []
        for v in input_dict.values():
            if isinstance(v, list):
                sub_out_list = v
            elif isinstance(v, dict):
                sub_out_list = LerobotFormatConvertor._find_lists_in_dict(v)
            out_list.extend(sub_out_list)

    @staticmethod
    def _validate_merged_features(merged_features: dict) -> None:
        feature_list = []
        LerobotFormatConvertor._find_lists_in_dict(merged_features, feature_list)

        cam_names: list[str] = []
        state_names: list[str] = []
        action_names: list[str] = []
        for feature_dict in feature_list:
            if LEROBOT_CAM_NAME_KEY in feature_dict:
                cam_names.append(feature_dict[LEROBOT_CAM_NAME_KEY])
            if LEROBOT_STATE_KEY in feature_dict:
                state_names.append(feature_dict[LEROBOT_STATE_KEY])
            if LEROBOT_ACTION_KEY in feature_dict:
                action_names.append(feature_dict[LEROBOT_ACTION_KEY])

        cam_counter = Counter(cam_names)
        duplicate_cams = [item for item, count in cam_counter.items() if count > 1]

        state_counter = Counter(cam_names)
        duplicate_states = [item for item, count in state_counter.items() if count > 1]

        action_counter = Counter(cam_names)
        duplicate_actions = [item for item, count in action_counter.items() if count > 1]

        if cam_counter or state_counter or action_counter:
            raise ValueError(
                "Duplicate names found in dataset. Please check the following: \n"
                f"duplicated images names: {duplicate_cams}\n"
                f"duplicated states names: {duplicate_states}\n"
                f"duplicated action names: {duplicate_actions}\n"
            )

    @staticmethod
    def _merge_and_validate_features(features: list[dict]) -> dict:
        """
        Merge multiple feature dictionaries into one.
        """
        merged_features = {}
        for feature_dict in features:
            merged_features = Merger(features).merge(merged_features, feature_dict)

        LerobotFormatConvertor._validate_merged_features(merged_features)
        return merged_features

    @staticmethod
    def _get_image_shape_from_np(image: np.ndarray) -> tuple:
        """
        Get the shape of the image from the feature dictionary.
        """

        height = 0
        width = 0
        channels = 0
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape[:2]
            channels = 1  # 对于灰度图
        return height, width, channels

    @cached_property
    def image_channel_names(self) -> list[str]:
        return LEROBOT_DEFAULT_IMAGE_SHAPE_NAMES

    def _add_episode(self, episode_id: int) -> None:
        episode_buffer_data = self._get_episode_buffer_data(episode_id)
        for frame_id in range(self._get_episode_frame_num(episode_id)):
            frame_buffer_data = self.
            self._add_episode_frame(episode_id, frame_id, episode_buffer_data)

    def convert(self) -> iter[int]:
        for episode_id in range(self._get_episode_num()):
            self._add_episode(episode_id)
            yield episode_id
