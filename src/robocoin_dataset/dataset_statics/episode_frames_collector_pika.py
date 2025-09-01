import os
from pathlib import Path

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorPika(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number

        def find_first_jpg_dir_count(episode_dir: str) -> int:
            for root, dirs, files in os.walk(episode_dir):
                jpg_files = [
                    f for f in files if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
                ]
                if len(jpg_files) > 0:
                    return len(jpg_files)
            return 0

        dirs_to_scan = []

        episode_dirs = []
        dirs_to_scan.append(self.dataset_dir)

        episode_frames_num: list[int] = []
        while len(dirs_to_scan) > 0:
            current_path = dirs_to_scan.pop(0)
            dirs = Path(current_path).iterdir()
            for dir in dirs:
                if not dir.is_dir():
                    continue
                if dir.name.startswith("episode"):
                    episode_dirs.append(dir)
                    continue
                dirs_to_scan.append(dir)
        counter = 0
        for episode_dir in episode_dirs:
            counter += 1
            episode_frame_num = find_first_jpg_dir_count(episode_dir)
            episode_frames_num.append(episode_frame_num)

        return episode_frames_num
