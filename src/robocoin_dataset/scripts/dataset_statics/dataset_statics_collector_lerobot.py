from pathlib import Path

import json_lines

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

        target_file = "meta/episodes.jsonl"
        if not self.dataset_dir.joinpath(target_file).exists():
            raise FileNotFoundError(f"File {target_file} not found in dataset directory.")
        with json_lines.open(target_file) as f:
            for line in f:
                self.dataset_statics.episode_frames_num.append(line["length"])
                continue
