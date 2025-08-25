from pathlib import Path

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorLerobot(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")

        episode_dirs = [
            d for d in self.dataset_dir.iterdir() if d.is_dir() and d.name.startswith("episode")
        ]

        for episode_dir in episode_dirs:
            camera_dirs = [
                d for d in episode_dir.iterdir() if d.is_dir() and d.name.startswith("camera")
            ]

            image_num = [len(list(d.iterdir())) for d in camera_dirs if not d.is_dir()]
            self.dataset_statics.episode_frames_num.append(image_num)
