"""
Constants used across the robocoin datasets module.

This module defines various constants, enums, and configuration values used
throughout the robocoin datasets functionality including file paths, patterns,
and platform-specific settings.
"""

from enum import Enum
from pathlib import Path


class DatasetsHubEnum(str, Enum):
    """
    Enumeration of supported dataset hub platforms.

    Attributes:
        HUGGINGFACE: Huggingface platform identifier.
        MODELSCOPE: Modelscope platform identifier.
    """

    huggingface = "huggingface"
    modelscope = "modelscope"


# Platform configuration
DS_PLATFORM_NAME = "robocoin"
"""str: Name of the dataset platform."""


# Upload configuration
DEFAULT_UPLOAD_ALLOW_PATTERNS = None
"""Optional[list[str]]: Default file patterns to allow during upload (None means all files)."""

DEFAULT_UPLOAD_IGNORE_PATTERNS = [".commit_message"]
"""list[str]: Default file patterns to ignore during upload."""


# Commit and versioning
COMMIT_MESSAGE_FILE = Path(".commit_message")
"""Path: Filename for storing commit messages."""

COMMIT_MSG_LABEL = "commit_msg"
"""str: Label for commit message in metadata."""

COMMIT_UUID_LABEL = "commit_uuid"
"""str: Label for local commit UUID in metadata."""

REMOTE_COMMIT_UUID_LABEL = "remote_commit_uuid"
"""str: Label for remote commit UUID in metadata."""

INIT_COMMIT_MSG = "initial_commit"
"""str: Default message for initial commits."""


MODELSCOPE_BUG_EXCEPTON_MSG = "Expecting value: line 1 column 1 (char 0)"
"""str: Known exception message from Modelscope API bug."""


DATASET_INFO_FILE = "dataset_info.yml"
"""str: Dataset information file name."""

LEROBOT_META_INFO_FILE = "meta/info.json"
"""str: LeRobot format metadata file path."""

LEROBOT_META_TASKS_FILE = "meta/tasks.jsonl"
"""str: LeRobot format tasks file path."""

ANNOTATION_SOURCE_FILE = "label/data_annotation.json"
"""str: Annotation data file path."""

DEVICE_INFO_SOURCE_FILE = "device/device_info.json"
"""str: Device information file path."""

README_FILE = "README.md"
"""str: README file name."""


LOCAL_DATASET_CHECK_STRUCTURE = [
    LEROBOT_META_INFO_FILE,
    ANNOTATION_SOURCE_FILE,
    DEVICE_INFO_SOURCE_FILE,
]
"""list[str]: Required files for basic local dataset validation."""

GEN_README_DATASET_ADDITIONAL_CHECK_STRUCTURE = [
    DATASET_INFO_FILE,
]
"""list[str]: Additional files required for README generation."""

UPLOAD_DATASET_ADDITIONAL_CHECK_STRUCTURE = [
    README_FILE,
]
"""list[str]: Additional files required for dataset upload."""

IGNORED_SUBTASKS = [
    "end",
    "abnormal",
]
"""list[str]: Subtasks that should be ignored during processing."""


HUB_DATASETS_INFO_FOLDER = "hub_datasets_infos"
"""str: Folder name for storing hub dataset information."""

DATASETS_UPLOAD_LOG_FOLDER = "datasets_upload_logs"
"""str: Folder name for upload log files."""

DATASETS_GEN_README_LOG_FOLDER = "datasets_genreadme_logs"
"""str: Folder name for README generation log files."""

DATASETS_GEN_INFO_LOG_FOLDER = "datasets_genreadme_logs"
"""str: Folder name for dataset info generation log files."""

DATASET_INFO_TEMPLATE_FILE = "templates/dataset_info.yml"
"""str: Path to the dataset info template file."""

DEFAULT_OUTPUT_LOG_PATH = Path("./outputs/logs")


DEVICE_LIST_KEY = "device_list"

DEVICE_TO_FEATURES_FILE = "device_lerobot/features/device2features.yml"


LEROBOT_FEATURES_KEY = "features"

LEROBOT_OBSERVATION_KEY = "observation"

LEROBOT_IMAGE_KEY = "images"

LEROBOT_STATE_KEY = "state"

LEROBOT_ACTION_KEY = "action"

FEATURE_CAM_NAME_KEY = "cam_name"

LEROBOT_FEATURE_DESCRIPTON_KEY = "description"

LEROBOT_DEFAULT_IMAGE_SHAPE_NAMES = ["height", "width", "channels"]

FEATURE_NAME_KEY = "names"

TASK_DESCRIPTIONS_KEY = "task_descriptions"



TASK_INDEX_KEY = "task_index"
