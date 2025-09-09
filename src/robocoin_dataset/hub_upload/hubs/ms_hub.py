import warnings
from pathlib import Path

from modelscope.hub.api import HubApi  # noqa: E402

from robocoin_dataset.hub_upload.constant import (  # noqa: E402
    DEFAULT_UPLOAD_ALLOW_PATTERNS,
    DEFAULT_UPLOAD_IGNORE_PATTERNS,
    MODELSCOPE_BUG_EXCEPTON_MSG,
)

from .abstract_hub import (  # noqa: E402
    AbstractUploadHub,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


class ModelscopeUploadHub(AbstractUploadHub):
    """
    Implementation of AbstractUploadHub for ModelScope dataset uploads.

    This class provides functionality to upload datasets to ModelScope Hub,
    including repository creation and file uploading with commit messages.
    Handles a known bug in ModelScope API by catching specific exception messages.

    Attributes:
        hub (HubApi): ModelScope API client instance.
    """

    from modelscope.hub.api import HubApi

    def __init__(self, token: str) -> None:
        """
        Initialize the ModelScope upload hub with authentication token.

        Args:
            token (str): Authentication token for ModelScope API.
        """
        super().__init__(token)
        self.hub = HubApi()

    def repo_exists(self, repo_id: str) -> bool:
        """
        Check if a dataset repository exists on ModelScope Hub.

        Args:
            repo_id (str): Identifier of the repository to check.

        Returns:
            bool: True if repository exists, False otherwise.
        """
        return self.hub.repo_exists(repo_id=repo_id, token=self.token, repo_type="dataset")

    def create_repo(self, repo_id: str) -> None:
        """
        Create a new dataset repository on ModelScope Hub.

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
        Upload a local folder to a ModelScope dataset repository.

        Args:
            folder_path (Path): Path to the local folder to upload.
            repo_id (str): Identifier of the target repository.
            commit_msg (str): Commit message for the upload.

        Returns:
            str: URL of the commit on ModelScope Hub, or success message if known bug occurs.

        Raises:
            Exception: If upload fails for any reason other than the known ModelScope bug.
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
        except Exception as e:
            if str(e) == MODELSCOPE_BUG_EXCEPTON_MSG:
                return f"Exception captured when Modelscope upload dataset {repo_id}, but the repo has been uploaded successfully."
            raise e

        return commit_info.commit_url
