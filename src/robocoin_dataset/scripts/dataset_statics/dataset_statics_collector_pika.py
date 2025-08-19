import os
from pathlib import Path

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorPika(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        statics_path = []
        for current_path, subdirs, files in os.walk(self.dataset_dir):
            if "statistic.txt" in files:
                file_path = Path(current_path) / "statistic.txt"
                statics_path.append(file_path)
                subdirs.clear()

        if not statics_path:
            raise FileNotFoundError(f"No statistic.txt found in {self.dataset_dir}")

        for file_path in statics_path:
            with open(file_path) as file:
                lines = file.readlines()
                for line in lines:
                    if "camera/color/pikaDepthCamera_l" in line:
                        parts = line.split(" ")
                        if len(parts) > 2:
                            self.dataset_statics.episode_frames_num.append(int(parts[2]))
                            break
