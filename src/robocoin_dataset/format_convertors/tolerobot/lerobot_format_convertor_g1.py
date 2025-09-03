import io
import json
import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from natsort import natsorted
from PIL import Image

from robocoin_dataset.format_convertors.tolerobot.constant import (
    ARGS_KEY,
    FEATURES_KEY,
    OBSERVATION_KEY,
    STATE_KEY,
    SUB_STATE_KEY,
)
from robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
)


@dataclass
class G1Buffer:
    g1_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


class LerobotFormatConvertorG1(LerobotFormatConvertor):
    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        convertor_config: dict,
        repo_id: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.g1_buffer: G1Buffer = G1Buffer()
        
        # G1多相机自动检测
        if self._has_auto_camera_detection(convertor_config):
            print("G1: 启用多相机自动检测")
            convertor_config = self._auto_configure_cameras(convertor_config, dataset_path)
        
        super().__init__(dataset_path, output_path, convertor_config, repo_id, logger)

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
            raise ValueError(f"Camera {camera_idx} not found. Available cameras: {list(camera_groups.keys())}")
        
        camera_files = camera_groups[camera_idx]
        
        if frame_idx >= len(camera_files):
            raise ValueError(f"Frame index {frame_idx} out of range for camera {camera_idx}. Available frames: {len(camera_files)}")
        
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
            sub_states_buffer = self._prepare_episode_state_buffer(task_path, ep_idx)
        
        # 解析JSON路径，例如: "states.left_arm.qpos"
        json_path = args_dict["json_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        
        # 获取帧数据
        frame_data = sub_states_buffer["data"][frame_idx]
        
        # 按路径导航到目标数据
        data = frame_data
        for path_part in json_path.split('.'):
            if path_part in data:
                data = data[path_part]
            else:
                raise ValueError(f"Path '{json_path}' not found in frame {frame_idx}")
        
        # 提取指定范围的数据
        if isinstance(data, list):
            return np.array(data[from_idx:to_idx])
        else:
            raise ValueError(f"Expected list data for path '{json_path}', got {type(data)}")

    # @override
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        names_num: int,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        if not sub_actions_buffer:
            sub_actions_buffer = self._prepare_episode_action_buffer(task_path, ep_idx)
        
        # 解析JSON路径，例如: "actions.left_arm.qpos"
        json_path = args_dict["json_path"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        
        # 获取帧数据
        frame_data = sub_actions_buffer["data"][frame_idx]
        
        # 按路径导航到目标数据
        data = frame_data
        for path_part in json_path.split('.'):
            if path_part in data:
                data = data[path_part]
            else:
                raise ValueError(f"Path '{json_path}' not found in frame {frame_idx}")
        
        # 提取指定范围的数据
        if isinstance(data, list):
            return np.array(data[from_idx:to_idx])
        else:
            raise ValueError(f"Expected list data for path '{json_path}', got {type(data)}")

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        json_file_path = self.task_episode_jsonfile_paths[task_path][ep_idx]
        try:
            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)
                return len(json_data["data"])
        except Exception as e:
            raise ValueError(f"Error while reading json file {json_file_path}: {e}")

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
            "json_data": self._get_episode_json_data(task_path, ep_idx)
        }

    # @override
    def _prepare_episode_state_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_json_data(task_path, ep_idx)

    # @override
    def _prepare_episode_action_buffer(self, task_path: Path, ep_idx: int) -> any:
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
        import re
        camera_groups = {}
        
        # G1相机文件名模式匹配：文件名最后一位数字表示相机索引
        camera_pattern = r'.*(\d)\.jpe?g$'  # 匹配文件名最后一位数字
        
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
        import re
        
        # 提取key中的数字
        match = re.search(r'(\d+)$', image_key)
        if match:
            return int(match.group(1))
        
        # 默认返回相机0
        return 0

    def _detect_cameras_from_images(self, jpg_files: list[Path]) -> list[str]:
        """
        从图像文件自动检测相机配置
        返回相机key列表：["color_0", "color_1", "color_2", ...]
        """
        camera_groups = self._group_images_by_camera_g1(jpg_files)
        camera_keys = []
        
        for camera_idx in sorted(camera_groups.keys()):
            camera_keys.append(f"color_{camera_idx}")
        
        return camera_keys

    def _has_auto_camera_detection(self, config: dict) -> bool:
        """检查配置中是否启用了多相机自动检测"""
        return (
            "camera_detection" in config and 
            config["camera_detection"].get("auto_detect", False)
        )

    def _auto_configure_cameras(self, config: dict, dataset_path: str) -> dict:
        """
        自动检测并配置多相机
        扫描数据集，检测相机数量并自动生成相机配置
        """
        print("正在扫描数据集以检测相机配置...")
        
        # 获取第一个任务的第一个episode来检测相机
        dataset_path_obj = Path(dataset_path)
        sample_images = []
        
        # 寻找样本图像文件
        for jpg_file in dataset_path_obj.rglob("*.jpg"):
            sample_images.append(jpg_file)
            if len(sample_images) >= 20:  # 取前20个文件作为样本
                break
        
        if not sample_images:
            print("未找到图像文件，使用默认单相机配置")
            return self._create_default_single_camera_config(config)
        
        # 检测相机
        detected_cameras = self._detect_cameras_from_images(sample_images)
        
        print(f"检测到 {len(detected_cameras)} 个相机: {detected_cameras}")
        
        # 生成相机配置
        return self._generate_multi_camera_config(config, detected_cameras)

    def _create_default_single_camera_config(self, config: dict) -> dict:
        """创建默认单相机配置"""
        config = config.copy()
        config["features"]["observation"]["images"] = [{
            "cam_name": "color_0",
            "args": {
                "image_key": "color_0"
            }
        }]
        return config

    def _generate_multi_camera_config(self, config: dict, camera_keys: list[str]) -> dict:
        """根据检测到的相机生成配置"""
        config = config.copy()
        
        # 生成相机图像配置
        images_config = []
        for camera_key in camera_keys:
            camera_config = {
                "cam_name": camera_key,
                "args": {
                    "image_key": camera_key
                }
            }
            images_config.append(camera_config)
        
        config["features"]["observation"]["images"] = images_config
        
        print(f"生成了 {len(images_config)} 个相机的配置:")
        for i, cam_config in enumerate(images_config):
            print(f"  相机 {i}: {cam_config['cam_name']}")
        
        return config

    def _find_episode_directories_g1(self, task_path: Path) -> list[Path]:
        """
        在JPG+JSON转换器中实现智能episode目录发现
        支持多种episode目录命名模式：episode1, episode_1, ep1, ep_1, 001, 1等
        支持深度搜索，找到分散在不同子目录中的episode
        """
        import re
        
        # Episode目录匹配模式
        episode_patterns = [
            r'^episode(\d+)$',      # episode1, episode2, episode10
            r'^episode_(\d+)$',     # episode_1, episode_2, episode_10  
            r'^ep(\d+)$',           # ep1, ep2, ep10
            r'^ep_(\d+)$',          # ep_1, ep_2, ep_10
            r'^(\d+)$',             # 1, 2, 10, 001, 002
        ]
        
        # 递归搜索所有子目录（限制深度避免过深搜索）
        def find_all_directories(root_path, max_depth=3, current_depth=0):
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
                relative_path = dir_path.relative_to(task_path) if dir_path.is_relative_to(task_path) else dir_path
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
                priority_files.append((1, f))
        
        # 检查包含"episode"的文件
        if not priority_files:
            for f in json_files:
                if "episode" in f.name.lower():
                    priority_files.append((2, f))
        
        # 按文件名长度排序
        if not priority_files:
            for f in json_files:
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
            with open(json_file_path, "r", encoding='utf-8') as json_file:
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
