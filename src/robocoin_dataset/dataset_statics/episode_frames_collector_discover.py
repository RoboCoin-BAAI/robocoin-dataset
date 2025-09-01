import os
from pathlib import Path

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorDiscover(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        camera_0_paths = []
        for current_path, subdirs, files in os.walk(self.dataset_dir):
            if "camera_0" in subdirs:
                camera_0_path = Path(current_path) / "camera_0"
                camera_0_paths.append(camera_0_path)
                subdirs.clear()

        if not camera_0_paths:
            raise FileNotFoundError(f"No camera_0 folder found in {self.dataset_dir}")

        episode_frames_num = []

        try:
            for camera_0_path in camera_0_paths:
                image_files = [f for f in os.listdir(camera_0_path) if f.lower().endswith(".jpg")]
                frame_count = len(image_files)
                episode_frames_num.append(frame_count)
        except Exception as e:
            raise Exception("Error while processing episode") from e

        return episode_frames_num
