from pathlib import Path

import h5py

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorHdf5(EpisodeFramesCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> list[int]:
        """
        Collect episode frames number from the dataset path.
        """
        # Implement the logic to collect episode frames number
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {self.dataset_dir} not found.")

        extensions = [".hdf5", ".h5"]
        hdf5_files: list[Path] = []
        for ext in extensions:
            hdf5_files.extend(self.dataset_dir.rglob(f"*{ext}"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.dataset_dir}")

        episode_frames_num: list[int] = []
        for hdf5_file in hdf5_files:
            if not hdf5_file.is_file():
                continue
            try:
                with h5py.File(hdf5_file, "r") as f:
                    episode_frames_num.append(f["/observations/qpos"].shape[0])
            except Exception:
                continue

        return episode_frames_num
