import importlib
import logging
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from robocoin_dataset.constant import (
    LOCAL_DATASET_INFO_FILE,
    LOCAL_TASK_INFO_FILE_NAME,
    TASK_DESCRIPTIONS_KEY,
    TASK_INDEX_KEY,
)
from robocoin_dataset.format_converter.tolerobot.constant import (
    ACTION_KEY,
    ARGS_KEY,
    CAM_NAME_KEY,
    CONVERT_FUNC_KEY,
    DEFAULT_IMAGE_SHAPE_NAMES,
    DTYPE_KEY,
    FEATURES_KEY,
    FLOAT32,
    FPS,
    FRAME_IDX_KEY,
    IMAGE_DTYPE_VALUE,
    IMAGE_KEY,
    LEROBOT_FEATURE_KEY,
    NAME_KEY,
    OBSERVATION_KEY,
    SHAPE_KEY,
    STATE_KEY,
    SUB_ACTION_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_converter.utils.spatial_data_convertor import spatial_covertor_funcs


class LerobotFormatConverter:
    """
    Base class for converting datasets to the LeRobot format.
    This class should be extended by specific dataset format converters.
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        if not dataset_path:
            raise ValueError("Dataset path must be provided.")
        if not output_path:
            raise ValueError("LeRobot destination path must be provided.")

        if dataset_path == output_path:
            raise ValueError("Dataset path and LeRobot destination path cannot be the same.")

        self.dataset_path = Path(dataset_path).expanduser().absolute()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")
        self.output_path = Path(output_path).expanduser().absolute()

        if not converter_config:
            raise ValueError("Convertor config must be provided.")

        self.converter_config = converter_config

        self.repo_id = repo_id

        if not logger:
            raise ValueError("Logger must be provided.")
        self.logger = logger
        self.logger.info(f"Using dataset path: {self.dataset_path}")
        self.device_model = device_model

        try:
            self._validate_convertor_config()
        except Exception as e:
            raise ValueError(f"Invalid features description: {e}") from e

        self.tasks = self._get_tasks()
        self.path_task_dict: dict[Path, str] = self._get_dataset_task_paths()
        self.task_episodes_num: dict[Path, int] = {}
        for task_path in self.path_task_dict.keys():
            self.task_episodes_num[task_path] = self._get_task_episodes_num(task_path)

        self.fps = self.converter_config[FPS]

        self._gen_image_configs()
        self._gen_action_configs()
        self._gen_state_configs()

        if not (self.dataset_path / LOCAL_DATASET_INFO_FILE).exists():
            raise ValueError(
                f"Dataset info file {LOCAL_DATASET_INFO_FILE} not found in {self.dataset_path}."
            )

        self.video_backend = video_backend
        self.image_writer_processes = image_writer_processes
        self.image_writer_threads = image_writer_threads

    @abstractmethod
    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        pass

    @abstractmethod
    def _get_task_episodes_num(self, task_path: Path) -> int:
        pass

    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        return None

    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        return None

    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        return None

    def _get_task_episodes_num(self, task_path: Path) -> int:
        if task_path not in self.task_episodes_num:
            return self.task_episodes_num[task_path]
        raise ValueError(f"Dataset task_path {task_path} not found")

    def _gen_episode_frame(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        images_buffer: any,
        states_buffer: any,
        actions_buffer: any,
    ) -> dict:
        """
        Generate a single frame for the episode.
        """
        frame_data = {}
        frame_data[FRAME_IDX_KEY] = frame_idx
        frame_data[IMAGE_KEY] = self._get_frame_images(task_path, ep_idx, frame_idx, images_buffer)
        frame_data[STATE_KEY] = self._get_frame_states(task_path, ep_idx, frame_idx, states_buffer)
        frame_data[ACTION_KEY] = self._get_frame_actions(
            task_path, ep_idx, frame_idx, actions_buffer
        )

        return frame_data

    def _validate_convertor_config(self) -> None:
        """
        Validate the convertor config.
        """
        # check features:
        if FEATURES_KEY not in self.converter_config:
            raise ValueError(f"Convertion config must contain {FEATURES_KEY} key.")

        # check features.observation:
        if OBSERVATION_KEY not in self.converter_config[FEATURES_KEY]:
            raise ValueError(
                f"Convertion config must contain {OBSERVATION_KEY} key in {FEATURES_KEY}."
            )

        # check features.observation.images:
        if IMAGE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY]:
            raise ValueError(
                f"Convertion config must contain {IMAGE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}."
            )

        cam_names: list[str] = []
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            if CAM_NAME_KEY not in image_config:
                raise ValueError(
                    f"Convertion config must contain {CAM_NAME_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}."
                )
            cam_names.append(image_config[CAM_NAME_KEY])
            if ARGS_KEY not in image_config:
                raise ValueError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}."
                )

        if len(set(cam_names)) != len(cam_names):
            raise ValueError(
                f"Convertion config has same cam_names in {FEATURES_KEY}.{OBSERVATION_KEY}.{IMAGE_KEY}"
            )

        if STATE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY]:
            raise ValueError(
                f"Convertion config must contain {STATE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}."
            )

        if SUB_STATE_KEY not in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY]:
            raise ValueError(
                f"Convertion config must contain {SUB_STATE_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}."
            )

        sub_state_names = []
        for sub_state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            if NAME_KEY not in sub_state_config:
                raise ValueError(
                    f"Convertion config must contain {NAME_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}.{SUB_STATE_KEY}."
                )
            sub_state_names.extend(sub_state_config[NAME_KEY])
            if ARGS_KEY not in sub_state_config:
                raise ValueError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}.{SUB_STATE_KEY}."
                )

        if len(set(sub_state_names)) != len(sub_state_names):
            raise ValueError(
                f"Convertion config has same state names in {FEATURES_KEY}.{OBSERVATION_KEY}.{STATE_KEY}"
            )

        # check features.action:
        if ACTION_KEY not in self.converter_config[FEATURES_KEY]:
            raise ValueError(f"Convertion config must contain {ACTION_KEY} key in {FEATURES_KEY}.")

        if SUB_ACTION_KEY not in self.converter_config[FEATURES_KEY][ACTION_KEY]:
            raise ValueError(
                f"Convertion config must contain {SUB_ACTION_KEY} key in {FEATURES_KEY}.{ACTION_KEY}."
            )

        sub_action_names = []
        for sub_action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            if NAME_KEY not in sub_action_config:
                raise ValueError(
                    f"Convertion config must contain {NAME_KEY} key in {FEATURES_KEY}.{ACTION_KEY}.{SUB_ACTION_KEY}."
                )
            sub_action_names.extend(sub_action_config[NAME_KEY])
            if ARGS_KEY not in sub_action_config:
                raise ValueError(
                    f"Convertion config must contain {ARGS_KEY} key in {FEATURES_KEY}.{ACTION_KEY}.{SUB_ACTION_KEY}."
                )

        if len(set(sub_action_names)) != len(sub_action_names):
            raise ValueError(
                f"Convertion config has same state names in {FEATURES_KEY}.{OBSERVATION_KEY}.{ACTION_KEY}"
            )

    def _get_tasks(self) -> list[str]:
        dataset_info_file_path = self.dataset_path / LOCAL_DATASET_INFO_FILE
        if not dataset_info_file_path.exists():
            raise ValueError(f"Dataset info file {dataset_info_file_path} not found in.")
        with open(dataset_info_file_path) as file:
            ds_info_dict = yaml.safe_load(file)
            if not ds_info_dict:
                raise ValueError(f"Dataset info file {dataset_info_file_path} is empty.")
            if TASK_DESCRIPTIONS_KEY not in ds_info_dict:
                raise ValueError(
                    f"Dataset info file {dataset_info_file_path} does not contain task_description."
                )
            if not isinstance(ds_info_dict[TASK_DESCRIPTIONS_KEY], list):
                raise ValueError(
                    f"Dataset info file {dataset_info_file_path} does not contain task_description as list[str]."
                )
            return ds_info_dict[TASK_DESCRIPTIONS_KEY]

    def _get_dataset_task_paths(self) -> dict[Path, str]:
        """Get the paths of all tasks in the dataset."""
        dirs_to_scan = [self.dataset_path]
        task_paths_dict = {}
        try:
            while len(dirs_to_scan) > 0:
                current_path = dirs_to_scan.pop(0)
                files = Path(current_path).glob(LOCAL_TASK_INFO_FILE_NAME)
                has_task_file = False
                for file in files:
                    with file.open("r") as f:
                        task_info_dict = yaml.safe_load(f)
                        task_index = task_info_dict[TASK_INDEX_KEY]
                        task = self.tasks[task_index]
                        task_paths_dict[file.parent] = task
                        has_task_file = True

                if not has_task_file:
                    sub_dirs = [item for item in Path(current_path).glob("*") if item.is_dir()]
                    dirs_to_scan.extend(sub_dirs)

        except Exception as e:
            raise e

        return task_paths_dict

    def _get_one_frame_image(self, args_dict: dict) -> np.ndarray:
        task_path = list(self.path_task_dict.keys())[0]
        return self._get_frame_image(
            task_path=task_path, ep_idx=0, frame_idx=0, args_dict=args_dict
        )

    def _get_one_frame_images(self, image_configs: list[dict]) -> dict[str, np.ndarray]:
        images = {}
        for image_config in image_configs:
            image_name = image_config[CAM_NAME_KEY]
            args_dict = image_config[ARGS_KEY]
            image = self._get_one_frame_image(args_dict)
            images[image_name] = image
        return images

    def _gen_image_configs(self) -> None:
        sample_frame_images = self._get_one_frame_images(
            self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]
        )
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            image_config[LEROBOT_FEATURE_KEY] = (
                f"{OBSERVATION_KEY}.{IMAGE_KEY}.{image_config[CAM_NAME_KEY]}"
            )
            try:
                image = sample_frame_images[image_config[CAM_NAME_KEY]]
                image_config[DTYPE_KEY] = IMAGE_DTYPE_VALUE
                image_config[NAME_KEY] = DEFAULT_IMAGE_SHAPE_NAMES
                image_config[SHAPE_KEY] = image.shape
            except KeyError as e:
                raise ValueError(
                    f"Convertion config has no {image_config[CAM_NAME_KEY]} in sample frame images"
                ) from e

    def _gen_state_configs(self) -> None:
        sub_state_names = []
        for sub_state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            sub_state_names.extend(sub_state_config[NAME_KEY])
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][NAME_KEY] = sub_state_names
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][SHAPE_KEY] = (
            len(sub_state_names),
        )
        self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][LEROBOT_FEATURE_KEY] = (
            f"{OBSERVATION_KEY}.{STATE_KEY}"
        )

    def _gen_action_configs(self) -> None:
        sub_action_names = []
        for sub_action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            sub_action_names.extend(sub_action_config[NAME_KEY])
        self.converter_config[FEATURES_KEY][ACTION_KEY][NAME_KEY] = sub_action_names

        self.converter_config[FEATURES_KEY][ACTION_KEY][SHAPE_KEY] = (len(sub_action_names),)

        self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY] = ACTION_KEY

    def _get_frame_images(
        self, task_path: Path, ep_idx: int, frame_idx: int, images_buffer: any
    ) -> dict[str, np.ndarray]:
        images = {}
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            lerobot_feature = image_config[LEROBOT_FEATURE_KEY]
            args_dict = image_config[ARGS_KEY]
            image = self._get_frame_image(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                args_dict=args_dict,
                images_buffer=images_buffer,
            )
            images[lerobot_feature] = image

        return images

    def _get_frame_states(
        self, task_path: Path, ep_idx: int, frame_idx: int, states_buffer: any = None
    ) -> dict[str, np.ndarray]:
        lerobot_feature = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            LEROBOT_FEATURE_KEY
        ]

        sub_states_datas: list[np.ndarray] = []
        for state_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SUB_STATE_KEY
        ]:
            args_dict = state_config[ARGS_KEY]
            sub_states_data = self._get_frame_sub_states(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                args_dict=args_dict,
                sub_states_buffer=states_buffer,
            )
            if CONVERT_FUNC_KEY in state_config:
                if state_config[CONVERT_FUNC_KEY] in spatial_covertor_funcs:
                    if spatial_covertor_funcs[state_config[CONVERT_FUNC_KEY]]:
                        sub_states_data = spatial_covertor_funcs[state_config[CONVERT_FUNC_KEY]](
                            sub_states_data
                        ).astype(np.float32)

            sub_states_datas.append(sub_states_data)

        return {lerobot_feature: np.concatenate(sub_states_datas)}

    def _get_frame_actions(
        self, task_path: Path, ep_idx: int, frame_idx: int, actions_buffer: any = None
    ) -> dict[str, np.ndarray]:
        lerobot_feature = self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY]
        sub_actions_datas: list[np.ndarray] = []
        for action_config in self.converter_config[FEATURES_KEY][ACTION_KEY][SUB_ACTION_KEY]:
            names_num = len(action_config[NAME_KEY])
            args_dict = action_config[ARGS_KEY]
            sub_actions_data = self._get_frame_sub_actions(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                args_dict=args_dict,
                names_num=names_num,
                sub_actions_buffer=actions_buffer,
            )

            if CONVERT_FUNC_KEY in action_config:
                if action_config[CONVERT_FUNC_KEY] in spatial_covertor_funcs:
                    if spatial_covertor_funcs[action_config[CONVERT_FUNC_KEY]]:
                        sub_actions_data = spatial_covertor_funcs[action_config[CONVERT_FUNC_KEY]](
                            sub_actions_data
                        ).astype(np.float32)
            sub_actions_datas.append(sub_actions_data)

        return {lerobot_feature: np.concatenate(sub_actions_datas)}

    def _get_lerobot_datas(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        images_buffer: any = None,
        states_buffer: any = None,
        actions_buffer: any = None,
    ) -> dict[str, np.ndarray]:
        return {
            **self._get_frame_images(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                images_buffer=images_buffer,
            ),
            **self._get_frame_states(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                states_buffer=states_buffer,
            ),
            **self._get_frame_actions(
                task_path=task_path,
                ep_idx=ep_idx,
                frame_idx=frame_idx,
                actions_buffer=actions_buffer,
            ),
        }

    def _prepare_episode_buffers(self, task_path: Path, ep_idx: int) -> tuple[any, any, any]:
        return (
            self._prepare_episode_images_buffer(task_path=task_path, ep_idx=ep_idx),
            self._prepare_episode_states_buffer(task_path=task_path, ep_idx=ep_idx),
            self._prepare_episode_actions_buffer(task_path=task_path, ep_idx=ep_idx),
        )

    def _get_lerobot_image_features(self) -> dict:
        lerobot_image_features = {}
        for image_config in self.converter_config[FEATURES_KEY][OBSERVATION_KEY][IMAGE_KEY]:
            leroot_feature_key = image_config[LEROBOT_FEATURE_KEY]
            image_feature = {}
            image_feature[DTYPE_KEY] = image_config[DTYPE_KEY]
            image_feature[SHAPE_KEY] = image_config[SHAPE_KEY]
            image_feature[NAME_KEY] = image_config[NAME_KEY]
            lerobot_image_features[leroot_feature_key] = image_feature
        return lerobot_image_features

    def _get_lerobot_state_feature(self) -> dict:
        lerobot_state_feature = {}
        lerobot_feature_key = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            LEROBOT_FEATURE_KEY
        ]
        state_feature = {}
        # state_feature[DTYPE_KEY] = self.convertor_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
        #     DTYPE_KEY
        # ]
        state_feature[DTYPE_KEY] = FLOAT32
        state_feature[SHAPE_KEY] = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            SHAPE_KEY
        ]
        state_feature[NAME_KEY] = self.converter_config[FEATURES_KEY][OBSERVATION_KEY][STATE_KEY][
            NAME_KEY
        ]
        lerobot_state_feature[lerobot_feature_key] = state_feature
        return lerobot_state_feature

    def _get_lerobot_action_feature(self) -> dict:
        lerobot_action_feature = {}
        lerobot_feature_key = self.converter_config[FEATURES_KEY][ACTION_KEY][LEROBOT_FEATURE_KEY]
        action_feature = {}
        action_feature[DTYPE_KEY] = FLOAT32
        action_feature[SHAPE_KEY] = self.converter_config[FEATURES_KEY][ACTION_KEY][SHAPE_KEY]
        action_feature[NAME_KEY] = self.converter_config[FEATURES_KEY][ACTION_KEY][NAME_KEY]
        lerobot_action_feature[lerobot_feature_key] = action_feature
        return lerobot_action_feature

    def _get_lerobot_features(self) -> dict:
        return {
            **self._get_lerobot_image_features(),
            **self._get_lerobot_state_feature(),
            **self._get_lerobot_action_feature(),
        }

    def _create_lerobot_dataset(self) -> LeRobotDataset:
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            features=self._get_lerobot_features(),
            fps=self.fps,
            robot_type=self.device_model,
            root=self.output_path,
            video_backend=self.video_backend,
            image_writer_processes=self.image_writer_processes,
            image_writer_threads=self.image_writer_threads,
        )

    def _get_episode_task(self, ep_idx: int) -> str:
        return ""

    def _gen_episode_frames(
        self,
        task_path: Path,
        ep_idx: int,
        images_buffer: any = None,
        states_buffer: any = None,
        actions_buffer: any = None,
    ) -> Iterable[dict]:
        for frame_idx in range(self._get_episode_frames_num(task_path=task_path, ep_idx=ep_idx)):
            try:
                frame_data = self._gen_episode_frame(
                    task_path, ep_idx, frame_idx, images_buffer, states_buffer, actions_buffer
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in episode {ep_idx} frame {frame_idx}: {e}")
                raise e
            yield frame_data

        pass

    def convert(self) -> Iterable[tuple[str, int, int]]:
        dataset = self._create_lerobot_dataset()
        ep_idx = 0
        for task_path, task in self.path_task_dict.items():
            for task_ep_idx in range(self._get_task_episodes_num(task_path)):
                images_buffer, states_buffer, actions_buffer = self._prepare_episode_buffers(
                    task_path, task_ep_idx
                )
                for frame_data in self._gen_episode_frames(
                    task_path, task_ep_idx, images_buffer, states_buffer, actions_buffer
                ):
                    lerobot_datas = self._get_lerobot_datas(
                        task_path=task_path,
                        ep_idx=task_ep_idx,
                        frame_idx=frame_data[FRAME_IDX_KEY],
                        images_buffer=images_buffer,
                        states_buffer=states_buffer,
                        actions_buffer=actions_buffer,
                    )
                    dataset.add_frame(frame=lerobot_datas, task=self._get_episode_task(task_ep_idx))

                dataset.save_episode()
                yield (task, task_ep_idx, ep_idx)
                ep_idx += 1

    def get_episodes_num(self) -> int:
        return sum(self._get_task_episodes_num(task) for task in self.path_task_dict.keys())


class LerobotFormatConverterFactory:
    @staticmethod
    def create_converter(
        dataset_path: Path,
        device_model: str,
        output_path: Path,
        converter_config: dict,
        converter_module_path: str,
        converter_class_name: str,
        repo_id: str,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        logger: logging.Logger | None = None,
        converter_log_dir: Path | None = None,
    ) -> LerobotFormatConverter:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        # Create logger from converter_log_dir if provided and logger is None
        if logger is None and converter_log_dir is not None:
            from robocoin_dataset.utils.logger import setup_logger
            logger = setup_logger(
                name="lerobot_format_converter", 
                log_dir=converter_log_dir, 
                level=logging.INFO
            )

        module = importlib.import_module(converter_module_path)
        convertor_class = getattr(module, converter_class_name)
        return convertor_class(
            dataset_path=dataset_path,
            output_path=output_path,
            converter_config=converter_config,
            repo_id=repo_id,
            device_model=device_model,
            logger=logger,
            video_backend=video_backend,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )
