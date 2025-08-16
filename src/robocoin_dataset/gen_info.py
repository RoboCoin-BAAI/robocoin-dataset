"""
RoboCoin Datasets Generate Dataset Readme.md from readme_template/readme.j2 template
usage:
python -m robocoin.datasets.gen_readme --config configs/upload.yaml
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import draccus
import yaml

from .constant import (
  ANNOTATION_SOURCE_FILE,
  DATASET_INFO_FILE,
  DATASET_INFO_TEMPLATE_FILE,
  IGNORED_SUBTASKS,
  LEROBOT_META_INFO_FILE,
  LEROBOT_META_TASKS_FILE,
)
from .local_datasets import LocalDsConfig, LocalDsUtil


@dataclass
class LocalDsInfoConfig(LocalDsConfig):
  """
  Configuration class for local dataset information generation.

  Attributes:
      task_tags_yamls_dir (str): Directory path containing task tags YAML files. Defaults to empty string.
      output_path (str): Path to the output directory for generated dataset info files. Defaults to empty string.
  """

  task_tags_yamls_dir: str = ""
  output_path: str = ""


class LocalDsInfoUtil(LocalDsUtil):
  """
  Utility class for generating dataset information files.

  This class generates dataset information files by extracting metadata from
  dataset files and combining it with template information.

  Attributes:
      config (LocalDsInfoConfig): Configuration object for the info generator.
      logger: Logger instance for the info generator.
  """

  def __init__(self, config: LocalDsInfoConfig) -> None:
    """
    Initialize the info generator with configuration.

    Args:
        config (LocalDsInfoConfig): Configuration object for the info generator.
    """
    super().__init__(config)
    self.config = config

    self.logger = self.setup_logger(logger_name="GEN_DATASET_INFO")

  @cached_property
  def info_template_file(self) -> Path:
    """
    Get the path to the dataset info template file with validation.

    Returns:
        Path: Absolute path to the info template file.

    Raises:
        FileNotFoundError: If the info template file does not exist.
    """
    path = Path(__file__).parent.resolve().joinpath(DATASET_INFO_TEMPLATE_FILE)
    if not path.exists():
      raise FileNotFoundError(f"info template file {path} does not exists")
    return path

  @cached_property
  def output_path(self) -> Path:
    """
    Get the output path for generated info files with validation.

    Returns:
        Path: Absolute path to the output directory.

    Raises:
        ValueError: If output_path is not set in configuration.
    """
    if not self.config.output_path:
      raise ValueError("Output path is not set.")
    output_path = Path(self.config.output_path).expanduser().absolute()
    if not output_path.exists():
      output_path.mkdir(parents=True, exist_ok=True)
    return output_path

  def _get_info_from_lerobot_meta(self, ds_name: str) -> tuple[int, int, str]:
    """
    Extract information from LeRobot metadata files.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        tuple[int, int, str]: Episodes count, frames count, and tasks list as string.

    Raises:
        FileNotFoundError: If metadata files do not exist.
        Exception: If there are errors reading the files.
    """
    info_file = self.root_path.joinpath(ds_name, LEROBOT_META_INFO_FILE)
    tasks_file = self.root_path.joinpath(ds_name, LEROBOT_META_TASKS_FILE)
    if not info_file.exists():
      raise FileNotFoundError(f"dataset meta info file {info_file} does not exists")

    if not tasks_file.exists():
      raise FileNotFoundError(f"dataset meta tasks file {tasks_file} does not exists")

    episodes_num = 0
    frames_num = 0
    tasks_list: list[str] = []
    tasks = ""

    try:
      with open(info_file, encoding="utf-8") as f:
        data = json.load(f)
        episodes_num = data.get("episodes_num", 0)
        frames_num = data.get("frames_num", 0)
    except Exception as e:
      raise e from e

    try:
      with open(info_file, encoding="utf-8") as f:
        data = json.load(f)
        episodes_num = data.get("episodes_num", 0)
        frames_num = data.get("frames_num", 0)
    except Exception as e:
      raise e from e

    with open(tasks_file, encoding="utf-8") as f:
      for line in f:
        if not line.strip():
          continue
        try:
          task_data = json.loads(line)
          tasks_list.append(task_data.get("task", ""))
        except json.JSONDecodeError as e:
          self.logger.error(f"Failed to decode JSON from line: {line.strip()}")
          raise e from e
      tasks = "\n".join(tasks_list)

    return episodes_num, frames_num, tasks

  def _get_subtasks_from_annotation(self, ds_name: str) -> str:
    """
    Extract subtasks from dataset annotation file.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        str: Subtasks list as string.

    Raises:
        FileNotFoundError: If annotation file does not exist.
    """
    ds_path = self.root_path.joinpath(ds_name, ANNOTATION_SOURCE_FILE)
    if not ds_path.exists():
      raise FileNotFoundError(f"dataset {ds_path} does not exists")

    sub_tasks_set = set()
    with open(ds_path, encoding="utf-8") as f:
      data = json.load(f)
      for item in data:
        annotation = item.get("annotation", {})
        video_labels = annotation.get("videoLabels", [])
        for video_label in video_labels:
          timeline_labels = video_label.get("timelinelabels", [])
          for timeline_label in timeline_labels:
            if timeline_label not in IGNORED_SUBTASKS:
              sub_tasks_set.add(timeline_label)

    return "\n".join(sub_tasks_set)

  def _generate_size_label(self, size: int) -> str:
    """
    Generate size category label based on frame count.

    Args:
        size (int): Number of frames.

    Returns:
        str: Size category label (e.g., "<1K", "1K-10K", etc.).
    """
    if size < 1000:
      return "<1K"
    if size < 10000:
      return "1K-10K"
    if size < 100000:
      return "10K-100K"
    if size < 1000000:
      return "100K-1M"
    if size < 10000000:
      return "1M-10M"
    if size < 100000000:
      return "10M-100M"
    if size < 1000000000:
      return "100M-1B"
    if size < 10000000000:
      return "1B-10B"
    if size < 100000000000:
      return "10B-100B"
    if size < 1000000000000:
      return "100B-1T"
    return ">1T"

  def _generate_tags(self, ds_name: str) -> list[str]:
    """
    Generate tags for a dataset from YAML tag files.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        list[str]: List of tags for the dataset.
    """
    if not self.config.task_tags_yamls_dir:
      self.logger.warning("Task tags directory path is not set, skipping tag generation.")
      return []
    tags_file_path = Path(self.config.task_tags_yamls_dir).joinpath(f"{ds_name}.yml")
    if not tags_file_path.exists():
      self.logger.warning(f"Tags file {tags_file_path} does not exist, skipping tag generation.")
      return []
    with open(tags_file_path, encoding="utf-8") as f:
      return yaml.safe_load(f)

  def _get_auto_generate_info(self, ds_name: str) -> dict:
    """
    Automatically generate dataset information from dataset files.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        dict: Dictionary containing auto-generated dataset information.
    """
    log_prefix = f"dataset {ds_name}:"
    try:
      self.check_dataset_dir_valid(ds_name=ds_name)
    except Exception as e:
      self.logger.error(f"{log_prefix} dataset {ds_name} is not valid: {e}")
      return {}

    episodes_num, frames_num, tasks = self._get_info_from_lerobot_meta(ds_name)
    sub_tasks = self._get_subtasks_from_annotation(ds_name)
    size_category = self._generate_size_label(frames_num)
    tags = self._generate_tags(ds_name)

    return {
      "episodes_num": episodes_num,
      "frames_num": frames_num,
      "tasks": tasks,
      "sub_tasks": sub_tasks,
      "size_categories": size_category,
      "dataset_tags": tags,
    }

  def _generate_info(self, ds_name: str) -> dict:
    """
    Generate complete dataset information by combining template and auto-generated data.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        dict: Complete dataset information dictionary.
    """
    if not self.info_template_file.exists():
      self.logger.error(
        f"dataset {ds_name}: info template file {self.info_template_file} does not exist"
      )
      return {}
    auto_generated_info: dict = self._get_auto_generate_info(ds_name=ds_name)
    info: dict = yaml.safe_load(self.info_template_file.read_text(encoding="utf-8"))
    info.update(auto_generated_info)

    return info

  def generate_infos(self) -> None:
    """
    Generate information files for all valid datasets in the root path.

    This method validates datasets, generates information for each one,
    and saves the information to YAML files in the output directory.
    """
    self.check_root_path_valid()

    if not self.info_template_file.exists():
      raise FileNotFoundError(f"info template file {self.info_template_file} not found")

    ds_names = self.get_root_path_subdirs()
    for ds_name in ds_names:
      log_prefix = f"dataset {ds_name}:"
      try:
        self.check_dataset_dir_valid(ds_name=ds_name)
      except Exception as e:
        self.logger.info(f"dataset {ds_name} is not a valid dataset: {e}")
        continue

      ds_info_file = self.output_path.joinpath(ds_name, DATASET_INFO_FILE)
      try:
        ds_info_file.parent.mkdir(parents=True, exist_ok=True)
        with open(
          ds_info_file,
          "w+",
          encoding="utf-8",
        ) as f:
          yaml.safe_dump(
            self._generate_info(ds_name=ds_name),
            f,
            allow_unicode=True,
            sort_keys=False,
          )
          self.logger.info(f"Generated info for '{ds_name}' at {ds_info_file}")
      except Exception as e:
        self.logger.error(f"{log_prefix} generate {DATASET_INFO_FILE} failed, {e}")

  pass


if __name__ == "__main__":
  """
    Main entry point for the dataset info generator.
    
    Parses command line configuration and runs the info generation process.
    """
  config = draccus.parse(LocalDsInfoConfig)
  generator = LocalDsInfoUtil(config)
  generator.generate_infos()
  pass
