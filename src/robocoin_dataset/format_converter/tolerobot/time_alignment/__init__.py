"""
ROS Bag 时间对齐模块

该模块提供了处理RosBag数据时间对齐的功能，包括：
- 应急时间对齐（截断到最短长度）
- 完整时间插值对齐
- 时间同步质量评估
"""

from .alignment_config import AlignmentConfig, AlignmentStrategy
from .alignment_strategies import (
    EmergencyTimeAligner,
    InterpolationTimeAligner,
    TimeAligner,
    create_time_aligner,
    emergency_align_topics,
    galaxea_align_topics,
    production_align_topics,
)
from .time_sync_analyzer import TimeSyncAnalyzer

__all__ = [
    "TimeAligner", 
    "EmergencyTimeAligner",
    "InterpolationTimeAligner", 
    "create_time_aligner",
    "emergency_align_topics",
    "production_align_topics",
    "galaxea_align_topics",
    "TimeSyncAnalyzer",
    "AlignmentConfig",
    "AlignmentStrategy"
]
