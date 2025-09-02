"""
RoboCoin Datasets Uploader
usage:
python -m robocoin.datasets.upload --config configs/upload.yaml
"""

import json
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import draccus
from tqdm import tqdm

from .constant import (
  COMMIT_MESSAGE_FILE,
  COMMIT_MSG_LABEL,
  COMMIT_UUID_LABEL,
  DS_PLATFORM_NAME,
  INIT_COMMIT_MSG,
  REMOTE_COMMIT_UUID_LABEL,
  UPLOAD_DATASET_ADDITIONAL_CHECK_STRUCTURE,
  DatasetsHubEnum,
)
from .local_datasets_util import LocalDsConfig, LocalDsUtil


@dataclass
class LocalDsUploadConfig(LocalDsConfig):
  """
  Configuration class for local dataset uploading.

  Attributes:
      hub_name (DatasetsHubEnum): Target hub platform for uploading datasets.
          Defaults to DatasetsHubEnum.HUGGINGFACE.
      token (str): Authentication token for the target hub platform. Defaults to empty string.
      output_path (str): Path to the output directory for commit history files. Defaults to empty string.
  """

  hub_name: DatasetsHubEnum = DatasetsHubEnum.huggingface
  token: str = ""
  output_path: str = ""


class LocalDsUploadUtil(LocalDsUtil):
  """
  Utility class for uploading local datasets to remote hubs.

  This class handles the process of validating local datasets, checking commit history,
  and uploading datasets to either Hugging Face or ModelScope platforms.

  Attributes:
      config (LocalDsUploadConfig): Configuration object for the uploader.
      hub: Hub-specific upload implementation (HuggingfaceUploadHub or ModelscopeUploadHub).
      logger: Logger instance for the uploader.
  """

  def __init__(self, config: LocalDsUploadConfig) -> None:
    """
    Initialize the uploader with configuration.

    Args:
        config (LocalDsUploadConfig): Configuration object for the uploader.

    Raises:
        ValueError: If the specified hub platform is not supported.
    """
    super().__init__(config)
    self.config = config

    if config.hub_name == DatasetsHubEnum.modelscope:
      from .hubs.ms_hub import ModelscopeUploadHub

      self.hub = ModelscopeUploadHub(self.config.token)
    elif config.hub_name == DatasetsHubEnum.huggingface:
      from .hubs.hf_hub import HuggingfaceUploadHub

      self.hub = HuggingfaceUploadHub(self.config.token)
    else:
      raise ValueError(f"hub {config.hub_name} is not supported.")
    pass

    self.logger = self.setup_logger(logger_name="UPLOAD_DATASETS")

  @cached_property
  def output_path(self) -> Path:
    """
    Get the output path for commit history files with validation.

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

  def _commit_msg_exists(self, ds_name: str) -> bool:
    """
    Check if commit message file exists for a dataset.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        bool: True if commit message file exists, False otherwise.
    """
    return self.root_path.joinpath(ds_name, COMMIT_MESSAGE_FILE).exists()

  def _get_or_init_commit_msg(self, ds_name: str) -> tuple[str, str]:
    """
    Get existing commit message or initialize a new one.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        tuple[str, str]: Commit message and commit UUID.

    Raises:
        JSONDecodeError: If commit message file contains invalid JSON.
        OSError: If there are file I/O errors.
    """
    commit_msg_file_path = self.root_path.joinpath(ds_name).joinpath(COMMIT_MESSAGE_FILE)
    commit_msg = INIT_COMMIT_MSG
    commit_uuid = uuid.uuid4()
    try:
      if self._commit_msg_exists(ds_name):
        with open(commit_msg_file_path, encoding="utf-8") as f:
          data = json.load(f)
          commit_msg = data.get(COMMIT_MSG_LABEL)
          commit_uuid = data.get(COMMIT_UUID_LABEL)
      else:
        with open(commit_msg_file_path, "w+", encoding="utf-8") as f:
          commit_msg = INIT_COMMIT_MSG
          commit_uuid = str(uuid.uuid4())
          json_data = {COMMIT_MSG_LABEL: commit_msg, COMMIT_UUID_LABEL: commit_uuid}
          json.dump(json_data, f, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, OSError) as e:
      raise e

    return commit_msg, commit_uuid

  def _commit_msg_history_file_path(self, ds_name: str) -> Path:
    """
    Get the path to the commit history file for a dataset.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        Path: Path to the commit history file.
    """
    return self.output_path.joinpath(f"{self.config.hub_name}", f"{ds_name}.json")

  def _get_commit_history_msg(self, ds_name: str) -> tuple[str, str]:
    """
    Get commit history message for a dataset.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        tuple[str, str]: Commit history message and commit history UUID.
    """
    commit_history_msg = ""
    commit_history_uuid = ""
    commit_msg_history_file_path = self._commit_msg_history_file_path(ds_name)
    if commit_msg_history_file_path.exists():
      with open(commit_msg_history_file_path, encoding="utf-8") as f:
        data = json.load(f)
        commit_history_msg = data.get(COMMIT_MSG_LABEL)
        commit_history_uuid = data.get(COMMIT_UUID_LABEL)

    return commit_history_msg, commit_history_uuid

  def _update_commit_history_msg(self, ds_name: str, remote_commit_uuid: str) -> None:
    """
    Update commit history message file with remote commit information.

    Args:
        ds_name (str): Name of the dataset.
        remote_commit_uuid (str): UUID of the remote commit.
    """
    commit_msg, commit_uuid = self._get_or_init_commit_msg(ds_name)
    commit_msg_history_file_path = self._commit_msg_history_file_path(ds_name)
    commit_msg_history_dir = commit_msg_history_file_path.parent
    commit_msg_history_dir.mkdir(parents=True, exist_ok=True)

    with open(commit_msg_history_file_path, "w+", encoding="utf-8") as f:
      json_data = {
        COMMIT_MSG_LABEL: commit_msg,
        COMMIT_UUID_LABEL: commit_uuid,
        REMOTE_COMMIT_UUID_LABEL: remote_commit_uuid,
      }
      json.dump(json_data, f, indent=2, ensure_ascii=False)

  def _should_upload(self, ds_name: str) -> tuple[bool, str]:
    """
    Determine if a dataset should be uploaded based on commit history.

    Args:
        ds_name (str): Name of the dataset.

    Returns:
        tuple[bool, str]: Whether to upload and the commit message.
    """
    should_upload = True
    commit_msg, commit_uuid = self._get_or_init_commit_msg(ds_name)
    _, commit_history_uuid = self._get_commit_history_msg(ds_name)
    if commit_history_uuid == "":
      should_upload = True
    else:
      should_upload = commit_uuid != commit_history_uuid
    return should_upload, commit_msg

  def _upload_dataset(self, ds_name: str, commit_msg: str) -> bool:
    """
    Upload a single dataset to the remote hub.

    Args:
        ds_name (str): Name of the dataset to upload.
        commit_msg (str): Commit message for the upload.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    log_prefix = f"dataset {ds_name}:"
    try:
      self.check_dataset_dir_valid(
        ds_name=ds_name, additional_check_list=UPLOAD_DATASET_ADDITIONAL_CHECK_STRUCTURE
      )
    except Exception as e:
      self.logger.error(f"{log_prefix} {e}")
      return False

    ds_path = self.root_path.joinpath(ds_name)
    if not ds_path.exists():
      raise FileNotFoundError(f"dataset path {ds_path} does not exist")

    repo_id = f"{DS_PLATFORM_NAME}/{ds_name}"

    try:
      if not self.hub.repo_exists(repo_id=repo_id):
        self.logger.info(
          f"{log_prefix} repo {repo_id} does not exists in {self.config.hub_name}, creating repo {repo_id}"
        )
        self.hub.create_repo(repo_id=repo_id)

      commit_url: str = self.hub.upload_repo(ds_path, repo_id, commit_msg)
      remote_commit_uuid = commit_url.split("/")[-1]
      self.logger.info(
        f"{log_prefix} repo {repo_id} has been uploaded successfully, commit_url is: {commit_url}"
      )

      self._update_commit_history_msg(ds_name, remote_commit_uuid)
      return True

    except Exception as e:
      self.logger.error(f"{log_prefix} {e}")

    return False

  def upload_datasets(self) -> None:
    """
    Upload all datasets that need to be updated.

    This method validates datasets, checks which ones need to be uploaded based on
    commit history, and performs the upload process for each dataset.
    """
    datasets_to_update: dict[str, str] = {}
    self.check_root_path_valid()

    ds_names = self.get_root_path_subdirs()
    valid_ds_names: list[str] = []
    for ds_name in ds_names:
      try:
        self.check_dataset_dir_valid(
          ds_name, additional_check_list=UPLOAD_DATASET_ADDITIONAL_CHECK_STRUCTURE
        )
        valid_ds_names.append(ds_name)
        should_upload, commit_msg = self._should_upload(ds_name=ds_name)
        if should_upload:
          datasets_to_update[ds_name] = commit_msg
      except Exception as e:  # noqa: PERF203
        self.logger.error(f"{ds_name} is not valid, error: {e}")

    self.logger.info(
      f"found {len(valid_ds_names)} in path {self.root_path}, {len(datasets_to_update)} datasets need to update."
    )

    if datasets_to_update:
      for ds_name, commit_msg in tqdm(
        datasets_to_update.items(),
        desc=f"Upload {DS_PLATFORM_NAME} datasets",
        bar_format="\033[32m{l_bar}{bar}\033[0m{r_bar}",
      ):
        self._upload_dataset(ds_name=ds_name, commit_msg=commit_msg)

  pass


if __name__ == "__main__":
  """
    Main entry point for the dataset uploader.
    
    Parses command line configuration and runs the upload process.
    """
  config = draccus.parse(LocalDsUploadConfig)
  uploader = LocalDsUploadUtil(config)
  uploader.upload_datasets()
  pass
