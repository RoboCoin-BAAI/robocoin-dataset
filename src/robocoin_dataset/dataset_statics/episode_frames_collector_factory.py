import importlib
from pathlib import Path

import yaml

from .episode_frames_collector import EpisodeFramesCollector


class EpisodeFramesCollectorFactory:
    """
    Factory class to create dataset statics collectors based on the type.
    """

    def __init__(
        self,
        config_file: Path = Path(__file__).parent
        / "configs"
        / "episode_frames_collector_factory_config.yaml",
    ) -> None:
        self.config_file = config_file
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        if self.config_file.suffix != ".yml" and self.config_file.suffix != ".yaml":
            raise ValueError(f"Config file {self.config_file} must be a YAML file.")

        try:
            with open(self.config_file) as file:
                self.collectors_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config file {self.config_file}") from e

    def create_collector(
        self, dataset_info_file: Path, dataset_dir: Path
    ) -> EpisodeFramesCollector:
        """
        Create a dataset statics collector based on the collector type.

        Args:
            collector_type (str): Type of the collector (e.g., 'pika', 'unitree', 'hdf5').
            dataset_info_file (Path): Path to the dataset info file.
            dataset_dir (Path): Path to the dataset directory.

        Returns:
            EpisodeFramesCollector: An instance of the appropriate collector.
        """

        if not dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset info file {dataset_info_file} not found.")
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} not found.")

        if dataset_info_file.suffix != ".yaml" and dataset_info_file.suffix != ".yml":
            raise ValueError(f"Dataset info file {dataset_info_file} must be a YAML file.")

        device_model = []
        try:
            with open(dataset_info_file) as file:
                data = yaml.safe_load(file)
                device_model = data.get("device_model", "")
        except Exception as e:
            raise ValueError(f"Error loading dataset info file {dataset_info_file}: {e}")

        if not device_model:
            raise ValueError(f"Device model not found in dataset info file {dataset_info_file}.")
        device_model = device_model[0]
        if device_model not in self.collectors_config:
            raise ValueError(
                f"Unsupported device model: {device_model}. Available models: {list(self.collectors_config.keys())}"
            )

        try:
            module_path = self.collectors_config[device_model]["module"]
            class_name = self.collectors_config[device_model]["class"]
            module = importlib.import_module(module_path)
            collector_class = getattr(module, class_name)
            return collector_class(dataset_info_file, dataset_dir)
        except Exception as e:
            raise ImportError(
                f"Failed to import collector class for device model {device_model}. Error: {e}"
            )
