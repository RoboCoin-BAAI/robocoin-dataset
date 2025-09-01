from pathlib import Path

from rosbags.rosbag1 import Reader

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorGalaxea(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")

        bag_files: list[Path] = list(self.dataset_dir.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No bag files found in {self.dataset_dir}")

        episode_frames_num: list[int] = []

        try:
            for bag_file in bag_files:
                with Reader(bag_file) as reader:
                    for connection in reader.connections:
                        connection
                        if (
                            connection.topic
                            == "/hdas/camera_head/left_raw/image_raw_color/compressed"
                        ):
                            episode_frames_num.append(connection.msgcount)
                            break
        except Exception as e:
            raise e

        return episode_frames_num
