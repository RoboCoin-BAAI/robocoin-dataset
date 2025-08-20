import os
from pathlib import Path
import re

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorUnitree(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        statics_paths = []
        for current_path, subdirs, files in os.walk(self.dataset_dir):
            if "data.json" in files:
                file_path = Path(current_path) / "data.json"
                statics_paths.append(file_path)
                subdirs.clear()

        if not statics_paths:
            raise FileNotFoundError(f"No statistic.txt found in {self.dataset_dir}")
        
        idx_pattern = re.compile(rb'"idx"\s*:\s*(\d+)')

        for file_path in statics_paths:
            idx_count = 0  
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        for m in idx_pattern.finditer(line):
                            idx_count += 1  
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

            self.dataset_statics.episode_frames_num.append(idx_count)
            

        


























