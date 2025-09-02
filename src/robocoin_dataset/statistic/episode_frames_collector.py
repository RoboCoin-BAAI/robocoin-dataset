from abc import abstractmethod
from pathlib import Path


class EpisodeFramesCollector:
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
        if self.dataset_info_file.suffix != ".yml" and self.dataset_info_file.suffix != ".yaml":
            raise ValueError(f"Dataset info file {self.dataset_info_file} must be a YAML file.")

        self.episode_frames_num = []

    def collect(self) -> list[int]:
        """
        Collect dataset episode frames nums.
        """
        return self._collect_episode_frames_num()

    @abstractmethod
    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        pass
