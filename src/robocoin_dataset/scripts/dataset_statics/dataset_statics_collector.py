from abc import abstractmethod
from pathlib import Path

import yaml

from .dataset_statics import DatasetStatics, TaskObject


class DatasetStaticsCollector:
    """
    Collect dataset statics for Robocoin dataset.
    """

    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        self.dataset_info_file = dataset_info_file
        self.dataset_dir = dataset_dir
        if not self.dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset info file {self.dataset_info_file} not found.")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")
        if self.dataset_info_file.suffix != ".yml":
            raise ValueError(f"Dataset info file {self.dataset_info_file} must be a YAML file.")

        self.dataset_statics = DatasetStatics()

    def collect(self) -> DatasetStatics:
        """
        Collect dataset statics.
        """
        self._collect_dataset_info()
        self._collect_episode_frames_num()
        pass

    def _collect_dataset_info(self) -> DatasetStatics:
        """
        Collect dataset info from the dataset path.
        """
        # Implement the logic to collect dataset info
        with open(
            self.dataset_info_file,
        ) as file:
            data = yaml.safe_load(file)
            self.dataset_statics.dataset_name = data.get("dataset_name", "")
            self.dataset_statics.dataset_path = str(self.dataset_dir)
            self.dataset_statics.device_model = data.get("device_model", "")
            self.dataset_statics.end_effector_type = data.get("end_effector_type", "")
            self.dataset_statics.atomic_actions = data.get("atomic_actions", [])
            self.dataset_statics.episode_frames_num = data.get("episode_frames_num", [])
            objects = data.get("objects", [])
            for obj in objects:
                task_object = TaskObject(
                    object_name=obj.get("object_name", ""),
                    level_category=obj.get("level_category", {}),
                )
                self.dataset_statics.objects.append(task_object)

        pass

    @abstractmethod
    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        pass
