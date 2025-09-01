from pathlib import Path

import json_lines

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorLerobot(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")
        episode_frames_num: list[int] = []
        target_file = self.dataset_dir / "meta/episodes.jsonl"
        if not target_file.exists():
            raise FileNotFoundError(f"File {str(target_file)} not found in dataset directory.")
        with json_lines.open(target_file) as f:
            for line in f:
                episode_frames_num.append(line["length"])
                continue

        return episode_frames_num
