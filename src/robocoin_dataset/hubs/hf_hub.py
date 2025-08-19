from pathlib import Path

from huggingface_hub import HfApi

from robocoin_dataset.constant import (
  DEFAULT_UPLOAD_ALLOW_PATTERNS,
  DEFAULT_UPLOAD_IGNORE_PATTERNS,
)

from .abstract_hub import AbstractUploadHub


class HuggingfaceUploadHub(AbstractUploadHub):
  """
  Implementation of AbstractUploadHub for Hugging Face dataset uploads.

  This class provides functionality to upload datasets to Hugging Face Hub,
  including repository creation and file uploading with commit messages.

  Attributes:
      logger_name (str): Name of the logger used for Hugging Face API.
      hub (HfApi): Hugging Face API client instance.
  """

  def __init__(self, token: str) -> None:
    """
    Initialize the Hugging Face upload hub with authentication token.

    Args:
        token (str): Authentication token for Hugging Face API.
    """
    super().__init__(token)
    self.hub = HfApi()

  def repo_exists(self, repo_id: str) -> bool:
    """
    Check if a dataset repository exists on Hugging Face Hub.

    Args:
        repo_id (str): Identifier of the repository to check.

    Returns:
        bool: True if repository exists, False otherwise.
    """
    return self.hub.repo_exists(repo_id=repo_id, token=self.token, repo_type="dataset")

  def create_repo(self, repo_id: str) -> None:
    """
    Create a new dataset repository on Hugging Face Hub.

    If the repository already exists, this method does nothing due to exist_ok=True.

    Args:
        repo_id (str): Identifier for the new repository.
    """
    if not self.repo_exists(repo_id=repo_id):
      self.hub.create_repo(
        repo_id=repo_id,
        token=self.token,
        repo_type="dataset",
        exist_ok=True,
      )

  def upload_repo(self, folder_path: Path, repo_id: str, commit_msg: str) -> str:
    """
    Upload a local folder to a Hugging Face dataset repository.

    Args:
        folder_path (Path): Path to the local folder to upload.
        repo_id (str): Identifier of the target repository.
        commit_msg (str): Commit message for the upload.

    Returns:
        str: URL of the commit on Hugging Face Hub.

    Raises:
        Exception: If upload fails for any reason.
    """
    try:
      commit_info = self.hub.upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        token=self.token,
        repo_type="dataset",
        allow_patterns=DEFAULT_UPLOAD_ALLOW_PATTERNS,
        ignore_patterns=DEFAULT_UPLOAD_IGNORE_PATTERNS,
        commit_message=commit_msg,
      )
      return commit_info.commit_url
    except Exception as e:
      print(e)
      raise e from e
