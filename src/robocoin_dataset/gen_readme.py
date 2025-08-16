"""
RoboCoin Datasets Generate Dataset Readme.md from readme_template/readme.j2 template
usage:
python -m robocoin.datasets.gen_readme --config configs/upload.yaml
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import draccus
import yaml
from jinja2 import Environment, FileSystemLoader

from .constant import (
  DATASET_INFO_FILE,
  LEROBOT_META_INFO_FILE,
  README_FILE,
)
from .local_datasets import LocalDsConfig, LocalDsUtil


@dataclass
class LocalDsReadmeConfig(LocalDsConfig):
  """
  Configuration class for local dataset README generation.

  Attributes:
      dataset_info_root_path (str): Root path containing dataset info files. Defaults to empty string.
  """

  dataset_info_root_path: str = ""


class LocalDsReadmeUtil(LocalDsUtil):
  """
  Utility class for generating dataset README files from Jinja2 templates.

  This class generates README.md files for datasets by combining dataset information
  with a Jinja2 template, producing formatted documentation for each dataset.

  Attributes:
      config (LocalDsReadmeConfig): Configuration object for the README generator.
      logger: Logger instance for the README generator.
  """

  def __init__(self, config: LocalDsReadmeConfig) -> None:
    """
    Initialize the README generator with configuration.

    Args:
        config (LocalDsReadmeConfig): Configuration object for the README generator.
    """
    super().__init__(config)
    self.config = config

    self.logger = self.setup_logger(logger_name="GEN_DATASET_README")

  @cached_property
  def readme_template_file(self) -> Path:
    """
    Get the path to the README template file with validation.

    Returns:
        Path: Absolute path to the README template file.

    Raises:
        FileNotFoundError: If the README template file does not exist.
    """
    path = Path(__file__).parent.resolve().joinpath("templates", "readme.j2")
    if not path.exists():
      raise FileNotFoundError(f"readme template file {path} does not exists")
    return path

  def _generate_readme(self, ds_name: str) -> None:
    """
    Generate README file for a specific dataset using Jinja2 template.

    Args:
        ds_name (str): Name of the dataset to generate README for.

    Raises:
        FileNotFoundError: If meta info file does not exist.
        RuntimeError: If there are errors during README generation.
    """

    def get_meta_info_content() -> str:
      """
      Get the content of the meta info file.

      Returns:
          str: Content of the meta info file.

      Raises:
          FileNotFoundError: If meta info file does not exist.
      """
      meta_info_file = self.root_path.joinpath(ds_name, LEROBOT_META_INFO_FILE)
      if not meta_info_file.exists():
        raise FileNotFoundError(f"Meta info file {meta_info_file} does not exist.")
      return meta_info_file.read_text(encoding="utf-8")

    ds_info_file = (
      Path(self.config.dataset_info_root_path)
      .joinpath(ds_name, DATASET_INFO_FILE)
      .expanduser()
      .absolute()
    )
    ds_info: dict
    try:
      with open(ds_info_file) as f:
        ds_info = yaml.safe_load(f)
    except Exception as e:
      raise RuntimeError(e) from e

    try:
      env = Environment(loader=FileSystemLoader(self.readme_template_file.parent))
      env.globals["get_meta_info_content"] = get_meta_info_content
      readme_content = env.get_template(self.readme_template_file.name).render(
        dataset_name=ds_name, **ds_info
      )

      ds_path = self.root_path.joinpath(ds_name)
      with open(ds_path.joinpath("README.md"), "w") as f:
        f.write(readme_content)
    except Exception as e:
      raise RuntimeError(e) from e

  def generate_readmes(self) -> None:
    """
    Generate README files for all valid datasets in the root path.

    This method validates datasets and generates README.md files for each one
    using the Jinja2 template and dataset information files.
    """
    self.check_root_path_valid()
    ds_names = self.get_root_path_subdirs()

    for ds_name in ds_names:
      self.check_dataset_dir_valid(ds_name=ds_name)
      log_prefix = f"dataset {ds_name}:"

      try:
        self._generate_readme(ds_name=ds_name)
        self.logger.info(f"{log_prefix} generate {README_FILE} successfully")
      except Exception as e:
        self.logger.error(f"{log_prefix} generate {README_FILE} failed, {e}")

  pass


if __name__ == "__main__":
  """
    Main entry point for the README generator.
    
    Parses command line configuration and runs the README generation process.
    """
  config = draccus.parse(LocalDsReadmeConfig)
  generator = LocalDsReadmeUtil(config)
  generator.generate_readmes()
  pass
