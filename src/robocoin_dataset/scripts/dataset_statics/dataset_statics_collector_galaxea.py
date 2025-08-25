from pathlib import Path

from rosbags.rosbag1 import Reader

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorGalaxea(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")

        bag_files: list[Path] = list(self.dataset_dir.glob("*.bag"))
        if not bag_files:
            raise FileNotFoundError(f"No bag files found in {self.dataset_dir}")

        try:
            for bag_file in bag_files:
                with Reader(bag_file) as reader:
                    episode_frames_num = 0
                    for connection, timestamp, rawdata in reader.messages():
                        episode_frames_num += 1
                    self.dataset_statics.episode_frames_num.append(episode_frames_num)
        except Exception as e:
            print(f"Error processing {bag_file.name}: {e}")
