"""
Local dataset util base class
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

from .constant import (
  LOCAL_DATASET_CHECK_STRUCTURE,
)
from .log_config import LogConfig


@dataclass
class LocalDsConfig:
  """
  Configuration class for local dataset operations.

  Attributes:
      root_path (Path): Root directory path for datasets. Defaults to empty Path.
      process_all (bool): Flag to indicate whether to process all subdirectories. Defaults to True.
      subdirs_to_process (list[str]): List of specific subdirectories to process. Defaults to empty list.
      subdirs_to_ignore (list[str]): List of subdirectories to ignore during processing. Defaults to empty list.
      log_config (LogConfig): Logging configuration object. Defaults to empty LogConfig.
  """

  root_path: Path = field(default_factory=Path)
  process_all: bool = True
  subdirs_to_process: list[str] = field(default_factory=list)
  subdirs_to_ignore: list[str] = field(default_factory=list)
  log_config: LogConfig = field(default_factory=LogConfig)


class LocalDsUtil:
  """
  Utility class for handling local dataset operations.

  This class provides methods for validating dataset directories,
  checking file structures, and setting up logging for dataset operations.

  Attributes:
      config (LocalDsConfig): Configuration object for local dataset operations.
  """

  def __init__(self, config: LocalDsConfig) -> None:
    """
    Initialize LocalDsUtil with configuration.

    Args:
        config (LocalDsConfig): Configuration object for local dataset operations.
    """
    self.config = config

  def setup_logger(self, logger_name: str) -> logging.Logger:
    """
    Set up and configure logger for dataset operations.

    Args:
        logger_name (str): Name for the logger instance.

    Returns:
        logging.Logger: Configured logger instance.

    Raises:
        ValueError: If log_dir is not specified in log_config.
    """
    if not self.config.log_config.log_dir:
      raise ValueError("log_dir is required in log_config")

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, self.config.log_config.log_level.upper()))

    log_dir = Path(self.config.log_config.log_dir).expanduser().absolute()
    log_dir.mkdir(parents=True, exist_ok=True)

    if logger.handlers:
      logger.handlers.clear()

    formatter = logging.Formatter(
      fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{logger_name}_{timestamp}.log"

    log_filepath = log_dir / log_filename

    if self.config.log_config.log_to_console:
      ch = logging.StreamHandler()
      ch.setLevel(getattr(logging, self.config.log_config.log_level.upper()))
      ch.setFormatter(formatter)
      logger.addHandler(ch)

    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setLevel(getattr(logging, self.config.log_config.log_level.upper()))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging is enabled, output to: {log_filepath}")

    return logger

  @cached_property
  def root_path(self) -> Path:
    """
    Get the absolute root path for datasets with validation.

    Returns:
        Path: Absolute path to the root dataset directory.

    Raises:
        FileNotFoundError: If the root path does not exist.
        NotADirectoryError: If the root path is not a directory.
    """
    path = Path(self.config.root_path).expanduser().absolute()
    if not path.exists():
      raise FileNotFoundError(f"root_path {path} does not exists")
    if not path.is_dir():
      raise NotADirectoryError(f"root_path {path} is not a directory")
    return path

  def get_root_path_subdirs(self) -> list[str]:
    """
    Get list of subdirectories in the root path based on configuration.

    Returns:
        list[str]: List of subdirectory names in the root path.
    """
    self.check_root_path_valid()

    sub_dirs: list[str] = []

    if self.config.process_all:
      sub_dirs = [
        p.name
        for p in self.root_path.iterdir()
        if p.is_dir() and p.name not in self.config.subdirs_to_ignore
      ]
    else:
      sub_dirs = [
        p
        for p in self.config.subdirs_to_process
        if (self.root_path / p).is_dir() and p not in self.config.subdirs_to_ignore
      ]
    return sub_dirs

  def check_dataset_dir_valid(self, ds_name: str, additional_check_list: list[str] = []) -> None:
    """
    Check if a dataset directory is valid and contains required files.

    Args:
        ds_name (str): Name of the dataset directory to check.
        additional_check_list (list[str], optional): Additional files to check for. Defaults to [].

    Raises:
        FileNotFoundError: If dataset directory doesn't exist or required files are missing.
        NotADirectoryError: If the dataset path is not a directory.
    """
    missing_files: list[Path] = []
    ds_path = self.root_path / ds_name
    if not ds_path.exists():
      raise FileNotFoundError(f"dataset {ds_name} does not exists in {self.root_path}")
    if not ds_path.is_dir():
      raise NotADirectoryError(f"dataset {ds_name} is not a directory in {self.root_path}")
    for item in LOCAL_DATASET_CHECK_STRUCTURE + additional_check_list:
      file_path = self.root_path.joinpath(ds_name).joinpath(item)
      if not file_path.exists():
        missing_files.append(file_path)

    if missing_files:
      missing_files_str = "\n".join(map(str, missing_files))
      raise FileNotFoundError(f"Dataset {ds_name} missing:\n {missing_files_str}")

  def check_root_path_valid(self) -> None:
    """
    Validate that the root path exists and is a directory.

    Raises:
        FileNotFoundError: If the root path does not exist.
        NotADirectoryError: If the root path is not a directory.
    """
    if not self.root_path.exists():
      raise FileNotFoundError(f"root_path {self.root_path} does not exists")
    if not self.root_path.is_dir():
      raise NotADirectoryError(f"root_path {self.root_path} is not a directory")
