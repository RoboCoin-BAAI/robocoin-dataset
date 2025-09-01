import json
from pathlib import Path

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorUnitree(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """
        # statics_paths = []
        data_files = self.dataset_dir.glob("**/data.json")
        episode_frames_num = []
        for data_file in data_files:
            try:
                with open(data_file) as file:
                    data = json.load(file)
                    episode_frame_num = len(data["data"])
                    episode_frames_num.append(episode_frame_num)
            except Exception:  # noqa: PERF203
                continue

        return episode_frames_num
