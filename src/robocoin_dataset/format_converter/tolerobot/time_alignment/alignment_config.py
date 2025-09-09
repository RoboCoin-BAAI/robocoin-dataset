"""
时间对齐配置类
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class AlignmentStrategy(Enum):
    """时间对齐策略枚举"""
    EMERGENCY = "emergency"  # 应急策略：截断到最短长度
    INTERPOLATION = "interpolation"  # 插值策略：基于时间插值对齐
    NEAREST_NEIGHBOR = "nearest_neighbor"  # 最近邻策略：选择最近时间戳


@dataclass
class AlignmentConfig:
    """时间对齐配置"""
    
    # 对齐策略
    strategy: AlignmentStrategy = AlignmentStrategy.EMERGENCY
    
    # 基准topic选择
    base_topic: Optional[str] = None  # 如果为None，自动选择
    base_topic_keywords: List[str] = None  # 基准topic关键词优先级
    
    # 时间容忍度设置
    max_time_diff_ms: float = 100.0  # 最大时间差容忍度(毫秒)
    warn_time_diff_ms: float = 50.0  # 发出警告的时间差阈值
    
    # 插值设置
    interpolation_method: str = "linear"  # 插值方法: linear, nearest, cubic
    extrapolation_enabled: bool = False  # 是否允许外推
    
    # 质量控制
    min_sync_quality: float = 0.8  # 最小同步质量要求
    drop_low_quality_episodes: bool = False  # 是否丢弃低质量episodes
    
    def __post_init__(self):
        """初始化后处理"""
        if self.base_topic_keywords is None:
            self.base_topic_keywords = [
                'rgb', 'color',  # RGB相机优先
                'image', 'camera',  # 通用图像
                'depth',  # 深度图像
                'compressed'  # 压缩图像
            ]
    
    @classmethod
    def emergency_config(cls) -> 'AlignmentConfig':
        """创建应急配置"""
        return cls(
            strategy=AlignmentStrategy.EMERGENCY,
            max_time_diff_ms=1000.0,  # 应急模式容忍度更高
            warn_time_diff_ms=500.0
        )
    
    @classmethod 
    def production_config(cls) -> 'AlignmentConfig':
        """创建生产配置"""
        return cls(
            strategy=AlignmentStrategy.INTERPOLATION,
            max_time_diff_ms=50.0,  # 生产模式要求更严格
            warn_time_diff_ms=20.0,
            min_sync_quality=0.9,
            drop_low_quality_episodes=True
        )
    
    @classmethod
    def galaxea_config(cls) -> 'AlignmentConfig':
        """为Galaxea机器人优化的配置"""
        return cls(
            strategy=AlignmentStrategy.INTERPOLATION,
            base_topic_keywords=[
                'camera_head', 'rgb', 'color',  # 头部相机优先
                'camera_wrist', 'compressed', 'image'  # 手腕相机次优先
            ],
            max_time_diff_ms=30.0,  # 机器人控制需要高精度
            warn_time_diff_ms=15.0,
            interpolation_method="linear",
            min_sync_quality=0.85
        )
