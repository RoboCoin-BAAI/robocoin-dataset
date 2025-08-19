from pathlib import Path

import h5py

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorHdf5(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
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

        try:
            with h5py.File(hdf5_files[0], "r") as f:
                episode_frames_num = f["action"].shape()[0]
                self.dataset_statics.episode_frames_num += episode_frames_num
        except Exception as e:
            raise ValueError(f"Error opening HDF5 file: {e}")
