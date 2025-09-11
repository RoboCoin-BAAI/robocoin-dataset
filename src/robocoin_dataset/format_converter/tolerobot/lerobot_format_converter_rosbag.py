import io
import json
import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from natsort import natsorted
from PIL import Image
from rosbags.highlevel import AnyReader

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)
from robocoin_dataset.format_converter.tolerobot.time_alignment import (
    AlignmentConfig,
    TimeSyncAnalyzer,
    create_time_aligner,
)


@dataclass
class RosbagBuffer:
    rosbag_data: dict | None = None
    task_path: Path | None = None
    ep_idx: int | None = None


class LerobotFormatConverterRosbag(LerobotFormatConverter):
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
        alignment_config: AlignmentConfig | None = None,
    ) -> None:
        self.rosbag_buffer: RosbagBuffer = RosbagBuffer()
        
        # 配置时间对齐策略
        if alignment_config is None:
            # 根据设备模型选择合适的对齐配置
            if device_model and "galaxea" in device_model.lower():
                alignment_config = AlignmentConfig.galaxea_config()
            else:
                # 默认使用应急策略以保证兼容性
                alignment_config = AlignmentConfig.emergency_config()
        self.alignment_config = alignment_config
        
        # 初始化时间对齐组件（使用临时logger，如果没有提供的话）
        temp_logger = logger if logger else logging.getLogger(__name__)
        self.time_aligner = create_time_aligner(alignment_config, temp_logger)
        self.sync_analyzer = TimeSyncAnalyzer(temp_logger)
        
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
        
        # 如果logger在super()调用后发生了变化，重新初始化组件
        if self.logger != temp_logger:
            self.time_aligner = create_time_aligner(alignment_config, self.logger)
            self.sync_analyzer = TimeSyncAnalyzer(self.logger)

    def _prevalidate_files(self) -> None:
        """验证ROS bag数据集文件结构"""
        for path in self.path_task_dict.keys():
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist")
            if path.is_file():
                raise ValueError(f"{path} is a file, expected directory")

            # 检查是否有.bag文件
            bag_files = list(path.glob("*.bag"))
            if not bag_files:
                raise ValueError(f"No .bag files found in {path}")
            
            # 验证bag文件可读性
            for bag_file in bag_files:
                if not bag_file.is_file():
                    raise ValueError(f"Bag file {bag_file} is not a valid file")
                # 可以添加更多ROS bag特定的验证

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

        topic_name = args_dict["topic_name"]
        print("rosbag_override _get_frame_image")
        print(f"topic_name: {topic_name}, keys: {images_buffer.keys()}")
        
        # 从rosbag数据中获取图像
        if topic_name in images_buffer and frame_idx < len(images_buffer[topic_name]):
            image_data = images_buffer[topic_name][frame_idx]
            
            # 图像数据现在是numpy数组了
            if isinstance(image_data, np.ndarray):
                return image_data
            raise ValueError(f"Expected numpy array, got {type(image_data)}")
        raise ValueError(f"No image data found for topic {topic_name} at frame {frame_idx}")

    # @override
    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        topic_name = args_dict["topic_name"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        
        if topic_name in sub_states_buffer and frame_idx < len(sub_states_buffer[topic_name]):
            state_data = sub_states_buffer[topic_name][frame_idx]
            if isinstance(state_data, np.ndarray):
                return state_data[from_idx:to_idx]
            # 转换为numpy数组再切片
            return np.array(state_data, dtype=np.float32)[from_idx:to_idx]
        raise ValueError(f"No state data found for topic {topic_name} at frame {frame_idx}")

    # @override
    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        topic_name = args_dict["topic_name"]
        from_idx = args_dict["range_from"]
        to_idx = args_dict["range_to"]
        
        if topic_name in sub_actions_buffer and frame_idx < len(sub_actions_buffer[topic_name]):
            action_data = sub_actions_buffer[topic_name][frame_idx]
            if isinstance(action_data, np.ndarray):
                return action_data[from_idx:to_idx]
            # 转换为numpy数组再切片
            return np.array(action_data, dtype=np.float32)[from_idx:to_idx]
        raise ValueError(f"No action data found for topic {topic_name} at frame {frame_idx}")

    # @override
    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        # 使用时间对齐器来确定episode的帧数
        rosbag_data = self._get_episode_rosbag_data(task_path, ep_idx)
        
        # rosbag_data现在已经是对齐后的数据，所有topic长度应该一致
        if not rosbag_data:
            return 0
            
        # 获取任意一个topic的长度作为帧数
        first_topic_data = next(iter(rosbag_data.values()))
        frame_count = len(first_topic_data) if first_topic_data else 0
        
        if self.logger:
            self.logger.info(f"Episode {ep_idx} 帧数: {frame_count}")
            
        return frame_count

    # @override
    def _get_task_episodes_num(self, task_path: Path) -> int:
        return len(self.task_episode_rosbagfile_paths[task_path])

    # @override
    def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_rosbag_data(task_path, ep_idx)

    # @override
    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_rosbag_data(task_path, ep_idx)

    # @override
    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> any:
        return self._get_episode_rosbag_data(task_path, ep_idx)

    @cached_property
    def task_episode_rosbagfile_paths(self) -> dict[Path, list[Path]]:
        task_episode_paths = {}
        for path in self.path_task_dict.keys():
            if path.exists():
                rosbag_files = natsorted(list(path.rglob("*.bag")))
                # rosbag_files.extend(natsorted(list(path.rglob("*.mcap"))))
                # rosbag_files.extend(natsorted(list(path.rglob("*.db3"))))  # ROS2 sqlite bags
                task_episode_paths[path] = rosbag_files
        return task_episode_paths

    def _get_bag_metadata(self, bag_file_path: Path) -> dict:
        """
        获取与bag文件相关联的JSON元数据
        
        Args:
            bag_file_path: bag文件路径
            
        Returns:
            dict: JSON元数据，如果文件不存在返回空字典
        """
        # 寻找对应的JSON文件
        # 假设JSON文件命名规则：bag_name + "_metadata.json" 或者同名但扩展名为.json
        possible_json_paths = [
            bag_file_path.with_suffix('.json'),
            bag_file_path.parent / f"{bag_file_path.stem}_metadata.json",
            bag_file_path.parent / f"{bag_file_path.name}_metadata.json"
        ]
        
        for json_path in possible_json_paths:
            if json_path.exists():
                try:
                    with open(json_path, encoding='utf-8') as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    if self.logger:
                        self.logger.warning(f"Failed to read metadata from {json_path}: {e}")
        
        return {}

    def _get_episode_rosbag_data(self, task_path: Path, ep_idx: int) -> any:
        should_load = self.rosbag_buffer.task_path != task_path or self.rosbag_buffer.ep_idx != ep_idx
        if not should_load:
            print(f"buffer is ok, skipping Loading episode {ep_idx} from {task_path}")
            return self.rosbag_buffer.rosbag_data
        
        rosbag_file_path = self.task_episode_rosbagfile_paths[task_path][ep_idx]
        
        # 第1步：读取rosbag文件并按topic组织原始消息
        topic_messages = self._read_raw_topic_messages(rosbag_file_path)
        
        # 第2步：分析时间同步质量（可选）
        if self.logger:
            sync_report = self.sync_analyzer.analyze_topic_messages(topic_messages)
            self.logger.info(f"时间同步质量: {sync_report.overall_quality_score:.3f}")
            self.logger.info(f"推荐策略: {sync_report.recommended_strategy}")
        
        # 第3步：使用时间对齐器对齐数据
        aligned_messages = self.time_aligner.align_topics(topic_messages)
        
        # 第4步：处理对齐后的消息，转换为最终格式
        processed_data = {}
        for topic_name, messages in aligned_messages.items():
            if messages:
                processed_data[topic_name] = self._process_aligned_messages(messages, topic_name)
        
        # 缓存结果
        self.rosbag_buffer.rosbag_data = processed_data
        self.rosbag_buffer.task_path = task_path
        self.rosbag_buffer.ep_idx = ep_idx
        
        print(f"rosbag_file loaded and aligned using {self.alignment_config.strategy.value}")
        print(f"Topics: {list(processed_data.keys())}")
        return processed_data
    
    def _read_raw_topic_messages(self, rosbag_file_path: Path) -> dict[str, list[dict]]:
        """读取rosbag文件的原始消息，按topic组织"""
        topic_messages = {}
        
        with AnyReader([rosbag_file_path]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                topic_name = connection.topic
                if topic_name not in topic_messages:
                    topic_messages[topic_name] = []
                
                # 解序列化消息
                msg = reader.deserialize(rawdata, connection.msgtype)
                topic_messages[topic_name].append({
                    'timestamp': timestamp,
                    'data': msg
                })
        
        return topic_messages
    
    def _process_aligned_messages(self, messages: list[Any], topic_name: str) -> list[Any]:
        """处理已对齐的消息列表，转换为最终格式"""
        if not messages:
            return []
        
        first_msg = messages[0]
        
        # 处理图像消息 - sensor_msgs/Image
        if hasattr(first_msg, 'data') and hasattr(first_msg, 'height') and hasattr(first_msg, 'width'):
            return [self._convert_ros_image(msg) for msg in messages]
        
        # 处理压缩图像消息 - sensor_msgs/CompressedImage
        if hasattr(first_msg, 'data') and hasattr(first_msg, 'format'):
            return [self._convert_compressed_image(msg) for msg in messages]
            
        # 处理关节状态消息 - sensor_msgs/JointState
        if hasattr(first_msg, 'name') and hasattr(first_msg, 'position'):
            return [np.array(msg.position, dtype=np.float32) for msg in messages]
            
        # 处理IMU消息 - sensor_msgs/Imu
        if hasattr(first_msg, 'linear_acceleration') and hasattr(first_msg, 'angular_velocity'):
            imu_data = []
            for msg in messages:
                linear_acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
                angular_vel = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
                imu_data.append(np.array(linear_acc + angular_vel, dtype=np.float32))
            return imu_data
            
        # 处理Twist消息 - geometry_msgs/TwistStamped
        if hasattr(first_msg, 'twist') and hasattr(first_msg.twist, 'linear'):
            twist_data = []
            for msg in messages:
                twist_msg = msg.twist if hasattr(msg, 'twist') else msg
                linear = [twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z]
                angular = [twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z]
                twist_data.append(np.array(linear + angular, dtype=np.float32))
            return twist_data
            
        # 处理数值数组消息 - std_msgs/Float64MultiArray等
        if hasattr(first_msg, 'data') and isinstance(first_msg.data, (list, tuple)):
            return [np.array(msg.data, dtype=np.float32) for msg in messages]
            
        # 处理几何消息 - geometry_msgs类型
        if hasattr(first_msg, 'pose') or hasattr(first_msg, 'position'):
            return [self._convert_geometry_msg(msg) for msg in messages]
            
        # 默认处理：尝试转换为numpy数组
        try:
            return [self._msg_to_array(msg) for msg in messages]
        except:  # noqa: E722
            # 如果无法转换，返回原始消息
            return messages
    
    def _convert_ros_image(self, img_msg) -> np.ndarray:  # noqa: ANN001
        """转换ROS图像消息为numpy数组"""
        # 将字节数据转换为numpy数组
        image_data = np.frombuffer(img_msg.data, dtype=np.uint8)
        
        # 根据编码格式重塑数组
        if img_msg.encoding == 'rgb8':
            return image_data.reshape((img_msg.height, img_msg.width, 3))
        if img_msg.encoding == 'bgr8':
            image_array = image_data.reshape((img_msg.height, img_msg.width, 3))
            return image_array[:, :, [2, 1, 0]]  # BGR to RGB
        if img_msg.encoding == 'mono8':
            return image_data.reshape((img_msg.height, img_msg.width))
        # 默认处理：假设为RGB
        return image_data.reshape((img_msg.height, img_msg.width, -1))
    
    def _convert_compressed_image(self, comp_img_msg) -> np.ndarray:  # noqa: ANN001
        """转换ROS压缩图像消息为numpy数组"""  
        image = Image.open(io.BytesIO(comp_img_msg.data))
        return np.array(image)
        
    def _convert_geometry_msg(self, geom_msg) -> np.ndarray:  # noqa: ANN001
        """转换几何消息为numpy数组"""
        if hasattr(geom_msg, 'pose'):
            pose = geom_msg.pose
            if hasattr(pose, 'pose'):  # PoseWithCovariance
                pose = pose.pose
            return np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ], dtype=np.float32)
        if hasattr(geom_msg, 'position'):
            pos = geom_msg.position
            return np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        return np.array([], dtype=np.float32)
            
    def _msg_to_array(self, msg) -> np.ndarray:  # noqa: ANN001
        """通用消息转数组函数"""
        if hasattr(msg, 'data'):
            if isinstance(msg.data, (list, tuple)):
                return np.array(msg.data, dtype=np.float32)
            return np.array([msg.data], dtype=np.float32)
        # 尝试提取所有数值字段
        values = []
        for attr_name in dir(msg):
            if not attr_name.startswith('_'):
                attr_value = getattr(msg, attr_name)
                if isinstance(attr_value, (int, float)):
                    values.append(attr_value)
        return np.array(values, dtype=np.float32) if values else np.array([], dtype=np.float32)
