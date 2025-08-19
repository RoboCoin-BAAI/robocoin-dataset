import logging
from abc import ABC, abstractmethod
from pathlib import Path


class AbstractUploadHub(ABC):
  """
  Abstract base class for dataset upload hub functionality.

  This class defines the interface for uploading datasets to a remote repository.
  Subclasses must implement the abstract methods to provide specific upload functionality.

  Attributes:
      token (str): Authentication token for accessing the hub service.
  """

  def __init__(self, token: str) -> None:
    """
    Initialize the upload hub with authentication token.

    Args:
        token (str): Authentication token for the hub service.
    """
    self.token = token

  def disable_hub_logger(self, logger: logging.Logger) -> None:
    """
    Disable logging for the hub by setting logger level to suppress all messages.

    Args:
        logger (logging.Logger): Logger instance to disable.
    """
    logger.setLevel(logging.CRITICAL + 1)

  @abstractmethod
  def repo_exists(self, repo_id: str) -> bool:
    """
    Check if a repository exists.

    Args:
        repo_id (str): Identifier of the repository to check.

    Returns:
        bool: True if repository exists, False otherwise.
    """
    pass

  @abstractmethod
  def create_repo(self, repo_id: str) -> None:
    """
    Create a new repository.

    Args:
        repo_id (str): Identifier for the new repository.
    """
    pass

  @abstractmethod
  def upload_repo(self, folder_path: Path, repo_id: str, commit_msg: str) -> str:
    """
    Upload a local folder to a remote repository.

    Args:
        folder_path (Path): Path to the local folder to upload.
        repo_id (str): Identifier of the target repository.
        commit_msg (str): Commit message for the upload.

    Returns:
        str: Result or identifier of the upload operation.
    """
    pass
