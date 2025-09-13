import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)


@dataclass
class Mmk2Buffer:
    """MMK2数据缓冲区"""

    main_bson_data: dict[str, list[dict]] = None
    hand_bson_data: list[dict] = None
    camera_groups: dict[str, list[Path]] = None
    task_path: Path = None
    ep_idx: int = None


def parse_bson_document(data: bytes, offset: int = 0) -> tuple[dict, int]:
    """解析单个BSON文档"""
    if offset + 4 > len(data):
        return None, offset

    # 读取文档大小
    doc_size = struct.unpack("<I", data[offset : offset + 4])[0]
    if offset + doc_size > len(data):
        return None, offset

    doc_data = data[offset : offset + doc_size]
    result = {}

    pos = 4  # 跳过文档大小

    while pos < len(doc_data) - 1:  # 最后一个字节是结束符0x00
        if pos >= len(doc_data):
            break

        # 读取字段类型
        field_type = doc_data[pos]
        pos += 1

        # 读取字段名
        name_end = doc_data.find(b"\x00", pos)
        if name_end == -1:
            break
        field_name = doc_data[pos:name_end].decode("utf-8", errors="ignore")
        pos = name_end + 1

        # 根据类型解析值
        if field_type == 0x01:  # double
            if pos + 8 <= len(doc_data):
                value = struct.unpack("<d", doc_data[pos : pos + 8])[0]
                pos += 8
                result[field_name] = value
        elif field_type == 0x02:  # string
            if pos + 4 <= len(doc_data):
                str_len = struct.unpack("<I", doc_data[pos : pos + 4])[0]
                pos += 4
                if pos + str_len <= len(doc_data):
                    value = doc_data[pos : pos + str_len - 1].decode("utf-8", errors="ignore")
                    pos += str_len
                    result[field_name] = value
        elif field_type == 0x03:  # document
            if pos + 4 <= len(doc_data):
                subdoc_size = struct.unpack("<I", doc_data[pos : pos + 4])[0]
                if pos + subdoc_size <= len(doc_data):
                    subdoc, _ = parse_bson_document(doc_data, pos)
                    if subdoc is not None:
                        result[field_name] = subdoc
                    pos += subdoc_size
        elif field_type == 0x04:  # array
            if pos + 4 <= len(doc_data):
                array_size = struct.unpack("<I", doc_data[pos : pos + 4])[0]
                if pos + array_size <= len(doc_data):
                    array_doc, _ = parse_bson_document(doc_data, pos)
                    if array_doc is not None:
                        # 将字典转换为列表（BSON数组以索引为键）
                        array_list = []
                        for i in range(len(array_doc)):
                            if str(i) in array_doc:
                                array_list.append(array_doc[str(i)])  # noqa: PERF401
                        result[field_name] = array_list
                    pos += array_size
        elif field_type == 0x08:  # boolean
            if pos + 1 <= len(doc_data):
                value = doc_data[pos] != 0
                pos += 1
                result[field_name] = value
        elif field_type == 0x10:  # int32
            if pos + 4 <= len(doc_data):
                value = struct.unpack("<i", doc_data[pos : pos + 4])[0]
                pos += 4
                result[field_name] = value
        elif field_type == 0x12:  # int64
            if pos + 8 <= len(doc_data):
                value = struct.unpack("<q", doc_data[pos : pos + 8])[0]
                pos += 8
                result[field_name] = value
        else:
            # 未知类型，跳过
            break

    return result, offset + doc_size


class LerobotFormatConverterMmk2(LerobotFormatConverter):
    """MMK2机器人转换器 - JPG+BSON格式"""

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        self.mmk2_buffer: Mmk2Buffer = Mmk2Buffer()
        super().__init__(
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

    def _prevalidate_files(self) -> None:
        """Validate MMK2 dataset files and structure."""
        for task_path in self.path_task_dict.keys():
            if not task_path.exists():
                raise FileNotFoundError(f"Task path does not exist: {task_path}")
            if not task_path.is_dir():
                raise ValueError(f"Task path is not a directory: {task_path}")
            
            # Check for required episode directories
            episode_dirs = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
            if not episode_dirs:
                self.logger.warning(f"No episode directories found in {task_path}")
            else:
                # Enhanced validation: validate each episode's internal structure
                for i, episode_dir in enumerate(episode_dirs):
                    self._validate_mmk2_episode_structure(episode_dir, i)
            
            # Validate each episode directory structure
            for episode_dir in episode_dirs:
                # Check for required subdirectories (observations, actions, etc.)
                required_subdirs = ['observations']
                for subdir in required_subdirs:
                    subdir_path = episode_dir / subdir
                    if not subdir_path.exists():
                        self.logger.warning(f"Missing required subdirectory: {subdir_path}")
                
                # Check for image files in observations
                obs_dir = episode_dir / 'observations'
                if obs_dir.exists():
                    image_files = list(obs_dir.glob("*.jpg")) + list(obs_dir.glob("*.png")) + list(obs_dir.glob("*.jpeg"))
                    if not image_files:
                        self.logger.warning(f"No image files found in {obs_dir}")

    def _validate_mmk2_episode_structure(self, episode_dir: Path, ep_idx: int) -> None:
        """Validate internal MMK2 episode structure against configuration"""
        try:
            if self.logger:
                self.logger.info(f"Validating MMK2 episode structure: {episode_dir}")
            
            # Validate BSON files structure
            main_bson_file = episode_dir / "episode_0.bson"
            hand_bson_file = episode_dir / "xhand_control_data.bson"

            # Check required BSON files exist
            if not main_bson_file.exists():
                if self.logger:
                    self.logger.warning(f"Missing main BSON file: {main_bson_file}")
                return
            
            if not hand_bson_file.exists():
                if self.logger:
                    self.logger.warning(f"Missing hand BSON file: {hand_bson_file}")
            
            # Validate main BSON structure
            self._validate_main_bson_structure(main_bson_file)
            
            # Validate hand BSON structure
            if hand_bson_file.exists():
                self._validate_hand_bson_structure(hand_bson_file)
            
            # Validate camera directories
            self._validate_camera_structure(episode_dir)
            
            if self.logger:
                self.logger.info(f"MMK2 episode structure validation completed for {episode_dir}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating MMK2 episode structure: {e}")
    
    def _validate_main_bson_structure(self, bson_file: Path) -> None:
        """Validate main BSON file structure against configuration"""
        try:
            with open(bson_file, "rb") as f:
                content = f.read()
            
            doc, _ = parse_bson_document(content, 0)
            if not doc or "data" not in doc:
                if self.logger:
                    self.logger.warning(f"Invalid main BSON structure in {bson_file}")
                return
            
            available_paths = set(doc["data"].keys())
            if self.logger:
                self.logger.info(f"Available main BSON paths: {sorted(available_paths)}")
            
            # Get expected paths from configuration
            expected_paths = set()
            if hasattr(self, 'converter_config') and self.converter_config:
                # Check state configuration
                state_config = self.converter_config.get('features', {}).get('observation', {}).get('state', {})
                if 'sub_state' in state_config:
                    for sub_state in state_config['sub_state']:
                        if 'args' in sub_state and sub_state['args'].get('bson_file') == 'episode_0.bson':
                            data_path = sub_state['args'].get('data_path', '').lstrip('/')
                            expected_paths.add(data_path)
                
                # Check action configuration
                action_config = self.converter_config.get('features', {}).get('action', {})
                if 'sub_action' in action_config:
                    for sub_action in action_config['sub_action']:
                        if 'args' in sub_action and sub_action['args'].get('bson_file') == 'episode_0.bson':
                            data_path = sub_action['args'].get('data_path', '').lstrip('/')
                            expected_paths.add(data_path)
            
            # Validate expected paths exist
            missing_paths = expected_paths - available_paths
            if missing_paths:
                if self.logger:
                    self.logger.warning(f"Missing main BSON paths: {sorted(missing_paths)}")
            
            found_paths = expected_paths & available_paths
            if self.logger:
                self.logger.info(f"Validated main BSON paths ({len(found_paths)}): {sorted(found_paths)}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating main BSON structure: {e}")
    
    def _validate_hand_bson_structure(self, bson_file: Path) -> None:
        """Validate hand BSON file structure against configuration"""
        try:
            with open(bson_file, "rb") as f:
                content = f.read()
            
            doc, _ = parse_bson_document(content, 0)
            if not doc or "frames" not in doc:
                if self.logger:
                    self.logger.warning(f"Invalid hand BSON structure in {bson_file}")
                return
            
            frames = doc["frames"]
            if not frames:
                if self.logger:
                    self.logger.warning(f"No frames found in hand BSON: {bson_file}")
                return
            
            # Check first frame structure
            first_frame = frames[0]
            available_paths = set()
            
            # Extract available observation paths
            if "observation" in first_frame:
                obs_data = first_frame["observation"]
                if "left_hand" in obs_data:
                    available_paths.add("observation.left_hand")
                if "right_hand" in obs_data:
                    available_paths.add("observation.right_hand")
            
            if self.logger:
                self.logger.info(f"Available hand BSON paths: {sorted(available_paths)}")
            
            # Get expected paths from configuration
            expected_paths = set()
            if hasattr(self, 'converter_config') and self.converter_config:
                # Check state configuration for hand data
                state_config = self.converter_config.get('features', {}).get('observation', {}).get('state', {})
                if 'sub_state' in state_config:
                    for sub_state in state_config['sub_state']:
                        if 'args' in sub_state and sub_state['args'].get('bson_file') == 'xhand_control_data.bson':
                            data_path = sub_state['args'].get('data_path', '')
                            expected_paths.add(data_path)
                
                # Check action configuration for hand data
                action_config = self.converter_config.get('features', {}).get('action', {})
                if 'sub_action' in action_config:
                    for sub_action in action_config['sub_action']:
                        if 'args' in sub_action and sub_action['args'].get('bson_file') == 'xhand_control_data.bson':
                            data_path = sub_action['args'].get('data_path', '')
                            expected_paths.add(data_path)
            
            # Validate expected paths exist
            missing_paths = expected_paths - available_paths
            if missing_paths:
                if self.logger:
                    self.logger.warning(f"Missing hand BSON paths: {sorted(missing_paths)}")
            
            found_paths = expected_paths & available_paths
            if self.logger:
                self.logger.info(f"Validated hand BSON paths ({len(found_paths)}): {sorted(found_paths)}")
                self.logger.info(f"Hand BSON contains {len(frames)} frames")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating hand BSON structure: {e}")
    
    def _validate_camera_structure(self, episode_dir: Path) -> None:
        """Validate camera directory structure against configuration"""
        try:
            # Get expected cameras from configuration
            expected_cameras = set()
            if hasattr(self, 'converter_config') and self.converter_config:
                images_config = self.converter_config.get('features', {}).get('observation', {}).get('images', [])
                for image_config in images_config:
                    if 'args' in image_config and 'camera_dir' in image_config['args']:
                        camera_dir = image_config['args']['camera_dir']
                        expected_cameras.add(camera_dir)
            
            # Check available cameras
            available_cameras = set()
            for camera_dir in ['camera_0', 'camera_1', 'camera_2']:
                camera_path = episode_dir / camera_dir
                if camera_path.exists():
                    available_cameras.add(camera_dir)
                    # Count images
                    jpg_files = list(camera_path.glob("*.jpg"))
                    if self.logger:
                        self.logger.info(f"Camera {camera_dir}: {len(jpg_files)} images")
            
            if self.logger:
                self.logger.info(f"Available cameras: {sorted(available_cameras)}")
            
            # Validate expected cameras exist
            missing_cameras = expected_cameras - available_cameras
            if missing_cameras:
                if self.logger:
                    self.logger.warning(f"Missing camera directories: {sorted(missing_cameras)}")
            
            found_cameras = expected_cameras & available_cameras
            if self.logger:
                self.logger.info(f"Validated cameras ({len(found_cameras)}): {sorted(found_cameras)}")
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating camera structure: {e}")

    # @override
    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        if not images_buffer:
            images_buffer = self._prepare_episode_images_buffer(task_path, ep_idx)

        camera_dir = args_dict["camera_dir"]

        if "camera_groups" not in images_buffer:
            raise ValueError("camera_groups not found in images_buffer")

        # Add available_cameras info to buffer if not present
        if "available_cameras" not in images_buffer:
            images_buffer["available_cameras"] = set(images_buffer["camera_groups"].keys())

        available_cameras = images_buffer["available_cameras"]
        if camera_dir not in available_cameras:
            error_msg = f"Camera {camera_dir} not found. Available cameras: {sorted(available_cameras)}"
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.info("Please check your configuration file to ensure camera_dir matches available cameras")
                self.logger.info("Or consider using a different camera configuration version")
                
                # 建议使用twocam版本
                if available_cameras == {"camera_0", "camera_2"}:
                    self.logger.info("Detected camera_0 and camera_2 only. Consider using 'twocam_version' configuration.")
            
            # 提供更详细的错误信息以便调试
            raise ValueError(
                f"{error_msg}. "
                f"This usually indicates a mismatch between the configuration file and actual dataset structure. "
                f"Please update the converter configuration to use available cameras: {sorted(available_cameras)}"
            )

        camera_files = images_buffer["camera_groups"][camera_dir]

        # 处理稀疏图像：如果请求的帧索引超出范围，使用最后一个可用的图像
        if frame_idx >= len(camera_files):
            if len(camera_files) == 0:
                raise ValueError(f"No images available for camera {camera_dir}")

            # 使用最后一个可用的图像
            self.logger.warning(
                f"Frame index {frame_idx} out of range for camera {camera_dir}. Available frames: {len(camera_files)}. Using last available frame."
            )
            image_path = camera_files[-1]
        else:
            image_path = camera_files[frame_idx]

        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        # 读取并返回图像
        img = Image.open(image_path)
        return np.array(img)

    # @override
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        if not sub_states_buffer:
            sub_states_buffer = self._prepare_episode_states_buffer(task_path, ep_idx)

        bson_file = args_dict["bson_file"]
        data_path = args_dict["data_path"]
        range_from = args_dict["range_from"]
        range_to = args_dict["range_to"]

        # 根据不同的BSON文件处理数据
        if bson_file == "episode_0.bson":
            # 主要关节数据
            field = args_dict.get("field", "pos")  # 默认使用pos字段

            if data_path not in sub_states_buffer["main_data"]:
                raise ValueError(f"Data path '{data_path}' not found in main BSON data")

            data_list = sub_states_buffer["main_data"][data_path]

            if frame_idx >= len(data_list):
                raise ValueError(
                    f"Frame index {frame_idx} out of range. Available: {len(data_list)}"
                )

            frame_data = data_list[frame_idx]
            if "data" not in frame_data or field not in frame_data["data"]:
                raise ValueError(f"Field '{field}' not found in frame data")

            values = frame_data["data"][field]
            return np.array(values[range_from:range_to], dtype=np.float32)

        if bson_file == "xhand_control_data.bson":
            # 手部数据
            hand_data = sub_states_buffer["hand_data"]

            if frame_idx >= len(hand_data):
                raise ValueError(
                    f"Frame index {frame_idx} out of range for hand data. Available: {len(hand_data)}"
                )

            frame_data = hand_data[frame_idx]

            # 解析data_path，例如: "observation.left_hand"
            path_parts = data_path.split(".")
            data = frame_data
            for part in path_parts:
                if part in data:
                    data = data[part]
                else:
                    raise ValueError(f"Path '{data_path}' not found in hand data")

            if isinstance(data, list):
                return np.array(data[range_from:range_to], dtype=np.float32)
            raise ValueError(f"Expected list data for path '{data_path}', got {type(data)}")

        raise ValueError(f"Unknown BSON file: {bson_file}")

    # @override
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        if not sub_actions_buffer:
            sub_actions_buffer = self._prepare_episode_actions_buffer(task_path, ep_idx)

        # 动作数据处理逻辑与状态数据类似
        return self._get_frame_sub_states(
            task_path, ep_idx, frame_idx, args_dict, sub_actions_buffer
        )

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        """获取episode的帧数 - 使用主BSON文件的帧数"""
        episode_dir = self._get_episode_directory(task_path, ep_idx)
        main_bson_file = episode_dir / "episode_0.bson"

        if not main_bson_file.exists():
            raise ValueError(f"Main BSON file not found: {main_bson_file}")

        try:
            with open(main_bson_file, "rb") as f:
                content = f.read()

            doc, _ = parse_bson_document(content, 0)
            if doc and "data" in doc:
                # 获取任一数据路径的长度作为帧数
                for value in doc["data"].values():
                    if isinstance(value, list):
                        return len(value)

            return 0
        except Exception as e:
            raise ValueError(f"Error reading main BSON file: {e}")

    # @override
    def _get_task_episodes_num(self, task_path: Path) -> int:
        """获取任务的episode数量"""
        episode_dirs = [
            d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("episode")
        ]
        return len(episode_dirs)

    # @override
    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        """准备图像缓冲区"""
        episode_dir = self._get_episode_directory(task_path, ep_idx)

        camera_groups = {}
        for camera_dir in ["camera_0", "camera_1", "camera_2"]:
            camera_path = episode_dir / camera_dir
            if camera_path.exists():
                jpg_files = natsorted(list(camera_path.glob("*.jpg")))
                camera_groups[camera_dir] = jpg_files

        # Add available cameras metadata for validation
        available_cameras = set(camera_groups.keys())
        
        return {
            "camera_groups": camera_groups,
            "available_cameras": available_cameras
        }

    # @override
    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        """准备状态数据缓冲区"""
        episode_dir = self._get_episode_directory(task_path, ep_idx)

        # 读取主BSON文件
        main_bson_file = episode_dir / "episode_0.bson"
        main_data = {}
        if main_bson_file.exists():
            with open(main_bson_file, "rb") as f:
                content = f.read()
            doc, _ = parse_bson_document(content, 0)
            if doc and "data" in doc:
                main_data = doc["data"]

        # 读取手部BSON文件
        hand_bson_file = episode_dir / "xhand_control_data.bson"
        hand_data = []
        if hand_bson_file.exists():
            with open(hand_bson_file, "rb") as f:
                content = f.read()
            doc, _ = parse_bson_document(content, 0)
            if doc and "frames" in doc:
                hand_data = doc["frames"]

        return {"main_data": main_data, "hand_data": hand_data}

    # @override
    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        """准备动作数据缓冲区 - 与状态数据共用"""
        return self._prepare_episode_states_buffer(task_path, ep_idx)

    def _get_episode_directory(self, task_path: Path, ep_idx: int) -> Path:
        """获取episode目录"""
        episode_dirs = sorted(
            [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith("episode")]
        )
        if ep_idx >= len(episode_dirs):
            raise ValueError(f"Episode index {ep_idx} out of range. Available: {len(episode_dirs)}")
        return episode_dirs[ep_idx]
