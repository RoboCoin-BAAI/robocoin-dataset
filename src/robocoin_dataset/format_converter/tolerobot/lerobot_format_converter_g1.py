import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)


@dataclass
class G1Buffer:
    g1_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


class LerobotFormatConverterG1(LerobotFormatConverter):
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
        self.g1_buffer: G1Buffer = G1Buffer()

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
        """Validate G1 dataset files and structure."""
        for task_path in self.path_task_dict.keys():
            if not task_path.exists():
                raise FileNotFoundError(f"Task path does not exist: {task_path}")
            if not task_path.is_dir():
                raise ValueError(f"Task path is not a directory: {task_path}")
            
            # Check for required JSON files and image directories
            episode_dirs = [d for d in task_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
            if not episode_dirs:
                self.logger.warning(f"No episode directories found in {task_path}")
            
            # Validate each episode directory
            for episode_dir in episode_dirs:
                # Check for JSON files
                json_files = list(episode_dir.glob("*.json"))
                if not json_files:
                    self.logger.warning(f"No JSON files found in {episode_dir}")
                
                # Check for image files
                image_files = list(episode_dir.glob("*.jpg")) + list(episode_dir.glob("*.png"))
                if not image_files:
                    self.logger.warning(f"No image files found in {episode_dir}")
        
        # 验证JSON文件内部结构与配置匹配
        self._validate_g1_json_structure()

    def _validate_g1_json_structure(self) -> None:
        """验证G1 JSON文件内部结构是否与YAML配置匹配"""
        if self.logger:
            self.logger.info("Validating G1 JSON structure...")
        
        # 收集所有需要验证的JSON路径
        required_json_paths = set()
        
        # 从状态配置中收集JSON路径
        if "observation" in self.converter_config.get("features", {}):
            observation_config = self.converter_config["features"]["observation"]
            if "state" in observation_config and "sub_state" in observation_config["state"]:
                for state_config in observation_config["state"]["sub_state"]:
                    if "args" in state_config and "json_path" in state_config["args"]:
                        required_json_paths.add(state_config["args"]["json_path"])
        
        # 从动作配置中收集JSON路径
        if "action" in self.converter_config.get("features", {}):
            action_config = self.converter_config["features"]["action"]
            if "sub_action" in action_config:
                for action_config_item in action_config["sub_action"]:
                    if "args" in action_config_item and "json_path" in action_config_item["args"]:
                        required_json_paths.add(action_config_item["args"]["json_path"])
        
        # 收集所有需要验证的图像键
        required_image_keys = set()
        if "observation" in self.converter_config.get("features", {}):
            observation_config = self.converter_config["features"]["observation"]
            if "images" in observation_config:
                for image_config in observation_config["images"]:
                    if "args" in image_config and "image_key" in image_config["args"]:
                        required_image_keys.add(image_config["args"]["image_key"])
        
        if not required_json_paths and not required_image_keys:
            if self.logger:
                self.logger.warning("No json_path or image_key found in configuration, skipping G1 structure validation")
            return
        
        # 验证每个任务路径下的JSON文件
        validation_errors = []
        for task_path in self.path_task_dict.keys():
            json_files = self.task_episode_jsonfile_paths.get(task_path, [])
            
            if not json_files:
                validation_errors.append(f"No JSON files found in task path: {task_path}")
                continue
            
            # 验证第一个JSON文件作为样本（假设同一任务下的JSON文件结构一致）
            sample_json_file = json_files[0]
            try:
                with open(sample_json_file, encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 验证JSON数据基本结构
                if "data" not in json_data:
                    validation_errors.append(f"Missing 'data' key in {sample_json_file}")
                    continue
                
                if not isinstance(json_data["data"], list) or len(json_data["data"]) == 0:
                    validation_errors.append(f"'data' should be a non-empty list in {sample_json_file}")
                    continue
                
                # 验证第一帧数据结构
                first_frame = json_data["data"][0]
                missing_paths = []
                invalid_paths = []
                
                # 验证JSON路径
                for required_path in required_json_paths:
                    try:
                        # 导航到目标数据
                        data = first_frame
                        path_parts = required_path.split(".")
                        for part in path_parts:
                            if isinstance(data, dict) and part in data:
                                data = data[part]
                            else:
                                missing_paths.append(required_path)
                                break
                        else:
                            # 检查数据是否为有效的列表/数组
                            if not isinstance(data, (list, tuple)):
                                invalid_paths.append(f"{required_path} (not a list/array, got {type(data).__name__})")
                            elif len(data) == 0:
                                invalid_paths.append(f"{required_path} (empty array)")
                    except Exception as e:
                        invalid_paths.append(f"{required_path} (error: {e})")
                
                # 验证图像文件匹配
                missing_image_keys = []
                if required_image_keys:
                    # 检查是否有对应的图像文件
                    episode_dir = sample_json_file.parent
                    image_files = list(episode_dir.rglob("*.jpg")) + list(episode_dir.rglob("*.jpeg")) + list(episode_dir.rglob("*.png"))
                    
                    # 使用现有的图像分组逻辑
                    camera_groups = self._group_images_by_camera_g1(image_files)
                    
                    for image_key in required_image_keys:
                        camera_idx = self._extract_camera_index_from_key(image_key)
                        if camera_idx not in camera_groups:
                            missing_image_keys.append(f"{image_key} (camera {camera_idx} not found)")
                        elif len(camera_groups[camera_idx]) == 0:
                            missing_image_keys.append(f"{image_key} (no images for camera {camera_idx})")
                
                if missing_paths:
                    validation_errors.append(
                        f"Missing required JSON paths in {sample_json_file}: {missing_paths}"
                    )
                
                if invalid_paths:
                    validation_errors.append(
                        f"Invalid JSON data structure in {sample_json_file}: {invalid_paths}"
                    )
                
                if missing_image_keys:
                    validation_errors.append(
                        f"Missing required image keys in {task_path}: {missing_image_keys}"
                    )
                
                if self.logger and not missing_paths and not invalid_paths and not missing_image_keys:
                    self.logger.info(f"G1 JSON structure validation passed for task: {task_path}")
                        
            except Exception as e:
                validation_errors.append(f"Error reading JSON file {sample_json_file}: {e}")
        
        if validation_errors:
            error_msg = "G1 JSON structure validation failed:\n" + "\n".join(validation_errors)
            raise ValueError(error_msg)
        
        if self.logger:
            self.logger.info("G1 JSON structure validation completed successfully")

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

        # 从缓存的图像列表中获取指定帧的图像
        image_key = args_dict.get("image_key", "color_0")  # 默认为color_0
        print("g1_override _get_frame_image")
        print(f"image_key: {image_key}, frame_idx: {frame_idx}")

        # G1多相机支持：从image_key解析相机索引
        camera_idx = self._extract_camera_index_from_key(image_key)

        # 获取对应相机的图像文件列表
        camera_groups = images_buffer.get("camera_groups", {})

        if camera_idx not in camera_groups:
            available_cameras = list(camera_groups.keys())
            error_msg = f"Camera {camera_idx} not found. Available cameras: {available_cameras}"
            
            if self.logger:
                self.logger.error(error_msg)
                if not available_cameras:
                    self.logger.error("No camera data found in the dataset. Please check if images exist and are properly formatted.")
                    self.logger.error("Expected G1 image naming pattern: *0.jpg, *1.jpg, etc. where the last digit indicates camera index")
                else:
                    self.logger.info(f"Please update your configuration to use one of the available cameras: {available_cameras}")
                    self.logger.info("Or check if your dataset has the expected camera data")
            
            raise ValueError(error_msg)

        camera_files = camera_groups[camera_idx]
        
        if not camera_files:
            error_msg = f"No image files found for camera {camera_idx}"
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error("This could indicate missing or corrupted image data in the dataset")
            raise ValueError(error_msg)

        # 查找与frame_idx匹配的图像文件，支持稀疏采样
        target_image_path = None
        for image_path in camera_files:
            # 从文件名提取帧号，例如：000379_color_0.jpg -> 379
            filename = image_path.stem  # 去掉扩展名
            frame_num_str = filename.split("_")[0]  # 取第一部分
            try:
                file_frame_idx = int(frame_num_str)
                if file_frame_idx == frame_idx:
                    target_image_path = image_path
                    break
            except ValueError:
                continue

        if target_image_path is None:
            # 如果找不到精确匹配的帧，使用最近的帧或最后一帧
            if camera_files:
                # 找到最接近的帧
                closest_file = None
                min_distance = float("inf")
                for image_path in camera_files:
                    filename = image_path.stem
                    frame_num_str = filename.split("_")[0]
                    try:
                        file_frame_idx = int(frame_num_str)
                        distance = abs(file_frame_idx - frame_idx)
                        if distance < min_distance:
                            min_distance = distance
                            closest_file = image_path
                    except ValueError:
                        continue
                target_image_path = closest_file

            if target_image_path is None:
                raise ValueError(
                    f"No suitable image found for frame {frame_idx} in camera {camera_idx}"
                )

        if not target_image_path.exists():
            raise ValueError(f"Image file not found: {target_image_path}")

        # 读取并返回图像
        img = Image.open(target_image_path)
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

        # 解析JSON路径，例如: "states.left_arm.qpos"
        json_path = args_dict["json_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]

        # 获取帧数据
        frame_data = sub_states_buffer["data"][frame_idx]

        # 按路径导航到目标数据
        data = frame_data
        for path_part in json_path.split("."):
            if path_part in data:
                data = data[path_part]
            else:
                raise ValueError(f"Path '{json_path}' not found in frame {frame_idx}")

        # 提取指定范围的数据
        if isinstance(data, list):
            return np.array(data[from_idx:to_idx], dtype=np.float32)
        raise ValueError(f"Expected list data for path '{json_path}', got {type(data)}")

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

        # 解析JSON路径，例如: "actions.left_arm.qpos"
        json_path = args_dict["json_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]

        # 获取帧数据
        frame_data = sub_actions_buffer["data"][frame_idx]

        # 按路径导航到目标数据
        data = frame_data
        for path_part in json_path.split("."):
            if path_part in data:
                data = data[path_part]
            else:
                raise ValueError(f"Path '{json_path}' not found in frame {frame_idx}")

        # 提取指定范围的数据
        if isinstance(data, list):
            return np.array(data[from_idx:to_idx], dtype=np.float32)
        raise ValueError(f"Expected list data for path '{json_path}', got {type(data)}")

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        json_file_path = self.task_episode_jsonfile_paths[task_path][ep_idx]
        try:
            with open(json_file_path) as json_file:
                json_data = json.load(json_file)
                return len(json_data["data"])
        except Exception as e:
            raise ValueError(f"Error while reading json file {json_file_path}") from e

    # @override
    def _get_task_episodes_num(self, task_path: Path) -> int:
        return len(self.task_episode_jsonfile_paths[task_path])

    # @override
    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        """
        准备episode图像缓存
        支持G1多相机自动检索：
        - 单相机：*0.jpg
        - 多相机：*0.jpg, *1.jpg, *2.jpg, *3.jpg等
        """
        # 获取JSON文件路径
        json_file_path = self.task_episode_jsonfile_paths[task_path][ep_idx]
        json_dir = json_file_path.parent

        # 从JSON同级目录递归搜索JPG文件
        jpg_files = list(json_dir.rglob("*.jpg"))
        jpg_files.extend(list(json_dir.rglob("*.JPG")))  # 支持大写扩展名
        jpg_files.extend(list(json_dir.rglob("*.jpeg")))  # 支持jpeg格式
        jpg_files.extend(list(json_dir.rglob("*.JPEG")))

        # G1多相机自动分组：按文件名模式分组
        camera_groups = self._group_images_by_camera_g1(jpg_files)

        print(f"Found {len(jpg_files)} image files in {json_dir}")
        print(f"Detected {len(camera_groups)} cameras")

        for cam_idx, cam_files in camera_groups.items():
            print(f"  Camera {cam_idx}: {len(cam_files)} images")
            if cam_files:
                print(f"    First: {cam_files[0].name}")
                print(f"    Last: {cam_files[-1].name}")

        return {
            "camera_groups": camera_groups,
            "all_image_files": jpg_files,
            "json_data": self._get_episode_json_data(task_path, ep_idx),
        }

    # @override
    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_json_data(task_path, ep_idx)

    # @override
    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_json_data(task_path, ep_idx)

    @cached_property
    def task_episode_jsonfile_paths(self) -> dict[Path, list[Path]]:
        task_episode_paths = {}
        for path in self.path_task_dict.keys():
            if path.exists():
                # 方法1: 尝试智能episode目录发现
                episode_dirs = self._find_episode_directories_g1(path)

                if episode_dirs:
                    # 找到了有规律的episode目录
                    json_files = []
                    for episode_dir in episode_dirs:
                        # 在每个episode目录中查找JSON文件
                        episode_json_files = list(episode_dir.rglob("*.json"))
                        if episode_json_files:
                            # 选择主要的数据文件
                            main_file = self._select_main_json_file(episode_json_files)
                            json_files.append(main_file)

                    if json_files:
                        task_episode_paths[path] = json_files
                        print(f"Found {len(json_files)} episodes using directory pattern in {path}")
                        continue

                # 方法2: 回退到递归搜索
                json_files = natsorted(list(path.rglob("*.json")))
                task_episode_paths[path] = json_files
                print(f"Found {len(json_files)} JSON files using recursive search in {path}")

        return task_episode_paths

    def _group_images_by_camera_g1(self, jpg_files: list[Path]) -> dict[int, list[Path]]:
        """
        G1多相机图像分组
        根据文件名模式将图像按相机分组：
        G1格式：*0.jpg, *1.jpg, *2.jpg, *3.jpg (文件名最后一位数字表示相机索引)
        """
        camera_groups = {}

        # G1相机文件名模式匹配：文件名最后一位数字表示相机索引
        camera_pattern = r".*(\d)\.jpe?g$"  # 匹配文件名最后一位数字

        for jpg_file in jpg_files:
            match = re.match(camera_pattern, jpg_file.name, re.IGNORECASE)
            if match:
                camera_idx = int(match.group(1))

                if camera_idx not in camera_groups:
                    camera_groups[camera_idx] = []

                camera_groups[camera_idx].append(jpg_file)

        # 对每个相机的图像按文件名排序
        for camera_idx in camera_groups:
            camera_groups[camera_idx] = natsorted(camera_groups[camera_idx], key=lambda x: x.name)

        # 按相机索引排序
        return dict(sorted(camera_groups.items()))

    def _extract_camera_index_from_key(self, image_key: str) -> int:
        """
        从image_key提取相机索引
        color_0 -> 0
        color_1 -> 1
        camera_2 -> 2
        """

        # 提取key中的数字
        match = re.search(r"(\d+)$", image_key)
        if match:
            return int(match.group(1))

        # 默认返回相机0
        return 0

    def _find_episode_directories_g1(self, task_path: Path) -> list[Path]:
        """
        在JPG+JSON转换器中实现智能episode目录发现
        支持多种episode目录命名模式：episode1, episode_1, ep1, ep_1, 001, 1等
        支持深度搜索，找到分散在不同子目录中的episode
        """

        # Episode目录匹配模式
        episode_patterns = [
            r"^episode(\d+)$",  # episode1, episode2, episode10
            r"^episode_(\d+)$",  # episode_1, episode_2, episode_10
            r"^ep(\d+)$",  # ep1, ep2, ep10
            r"^ep_(\d+)$",  # ep_1, ep_2, ep_10
            r"^(\d+)$",  # 1, 2, 10, 001, 002
        ]

        # 递归搜索所有子目录（限制深度避免过深搜索）
        def find_all_directories(root_path, max_depth=3, current_depth=0):  # noqa: ANN001, ANN202
            dirs = []
            if current_depth >= max_depth:
                return dirs

            try:
                for item in root_path.iterdir():
                    if item.is_dir():
                        dirs.append(item)
                        # 递归搜索子目录
                        dirs.extend(find_all_directories(item, max_depth, current_depth + 1))
            except (PermissionError, OSError):
                pass  # 忽略权限或其他访问错误

            return dirs

        all_dirs = find_all_directories(task_path)
        episode_dirs = []

        for directory in all_dirs:
            dir_name = directory.name

            # 检查是否匹配任何episode模式
            for pattern in episode_patterns:
                match = re.match(pattern, dir_name, re.IGNORECASE)
                if match:
                    episode_num = int(match.group(1))

                    # 检查目录中是否包含JSON文件（确保是有效的episode目录）
                    json_files = list(directory.rglob("*.json"))
                    if json_files:
                        episode_dirs.append((episode_num, directory))
                    break

        if episode_dirs:
            # 按episode编号排序，然后按目录名排序（处理相同编号的情况）
            episode_dirs.sort(key=lambda x: (x[0], x[1].name))
            sorted_dirs = [d[1] for d in episode_dirs]

            print(f"Found episode directories in {task_path}:")
            for i, dir_path in enumerate(sorted_dirs):
                relative_path = (
                    dir_path.relative_to(task_path)
                    if dir_path.is_relative_to(task_path)
                    else dir_path
                )
                print(f"  Episode {i}: {dir_path.name} at {relative_path}")

            return sorted_dirs

        return []

    def _select_main_json_file(self, json_files: list[Path]) -> Path:
        """
        从多个JSON文件中选择主要的数据文件
        优先选择包含episode数据的文件
        """
        if len(json_files) == 1:
            return json_files[0]

        # 优先级规则：
        # 1. 文件名包含"data"的文件
        # 2. 文件名包含"episode"的文件
        # 3. 文件名最短的文件（通常是主文件）
        # 4. 按字母顺序第一个文件

        priority_files = []

        # 检查包含"data"的文件
        for f in json_files:
            if "data" in f.name.lower():
                priority_files.append((1, f))  # noqa: PERF401
        
                priority_files.append((1, f))

        # 检查包含"episode"的文件
        if not priority_files:
            for f in json_files:
                if "episode" in f.name.lower():
                    priority_files.append((2, f))  # noqa: PERF401
        
                    priority_files.append((2, f))

        # 按文件名长度排序
        if not priority_files:
            for f in json_files:
                priority_files.append((len(f.name), f))  # noqa: PERF401
        
                priority_files.append((len(f.name), f))

        # 选择优先级最高的文件
        priority_files.sort(key=lambda x: x[0])
        selected_file = priority_files[0][1]

        print(f"Selected main JSON file: {selected_file.name}")
        return selected_file

    def _get_episode_json_data(self, task_path: Path, ep_idx: int) -> any:
        should_load = self.g1_buffer.task_path != task_path or self.g1_buffer.ep_idx != ep_idx
        if not should_load:
            print(f"buffer is ok, skipping Loading episode {ep_idx} from {task_path}")
            return self.g1_buffer.g1_data

        json_file_path = self.task_episode_jsonfile_paths[task_path][ep_idx]
        try:
            with open(json_file_path, encoding='utf-8') as json_file:
                json_data = json.load(json_file)

                # 缓存JSON数据
                self.g1_buffer.g1_data = json_data
                self.g1_buffer.task_path = task_path
                self.g1_buffer.ep_idx = ep_idx

                print("json_file loaded")
                print(f"Total frames: {len(json_data.get('data', []))}")
                return json_data
        except Exception as e:
            raise ValueError(f"Error while reading json file {json_file_path}: {e}")
