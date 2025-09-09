"""
时间对齐策略实现
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from .alignment_config import AlignmentConfig, AlignmentStrategy
from .time_sync_analyzer import TimeSyncAnalyzer, SyncQualityReport


class TimeAligner(ABC):
    """时间对齐器基类"""
    
    def __init__(self, config: AlignmentConfig, logger=None):
        self.config = config
        self.logger = logger
        self.analyzer = TimeSyncAnalyzer(logger)
    
    @abstractmethod
    def align_topics(self, topic_messages: Dict[str, List[Dict]]) -> Dict[str, List[Any]]:
        """
        对齐topics到统一时间线
        
        Args:
            topic_messages: {topic_name: [{'timestamp': xxx, 'data': xxx}, ...]}
            
        Returns:
            aligned_data: {topic_name: [data1, data2, ...]}
        """
        pass
    
    def analyze_sync_quality(self, topic_messages: Dict[str, List[Dict]]) -> SyncQualityReport:
        """分析时间同步质量"""
        return self.analyzer.analyze_topic_messages(topic_messages)
    
    def _select_base_topic(self, topic_messages: Dict[str, List[Dict]]) -> str:
        """选择基准topic"""
        if self.config.base_topic and self.config.base_topic in topic_messages:
            return self.config.base_topic
        
        # 自动选择基准topic
        report = self.analyze_sync_quality(topic_messages)
        if report.base_topic_candidates:
            return report.base_topic_candidates[0]
        
        # 如果没有候选，选择消息数量最少的topic
        return min(topic_messages.keys(), key=lambda k: len(topic_messages[k]))
    
    def _log_info(self, message: str):
        """记录信息"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"[INFO] {message}")
    
    def _log_warning(self, message: str):
        """记录警告"""
        if self.logger:
            self.logger.warning(message)
        else:
            print(f"[WARNING] {message}")


class EmergencyTimeAligner(TimeAligner):
    """应急时间对齐器 - 截断到最短长度"""
    
    def align_topics(self, topic_messages: Dict[str, List[Dict]]) -> Dict[str, List[Any]]:
        """截断所有topic到最短长度"""
        if not topic_messages:
            return {}
        
        # 找到最短长度
        min_length = min(len(messages) for messages in topic_messages.values())
        
        # 记录统计信息
        total_messages = sum(len(messages) for messages in topic_messages.values())
        preserved_messages = min_length * len(topic_messages)
        data_loss = (total_messages - preserved_messages) / total_messages if total_messages > 0 else 0
        
        self._log_info(f"应急对齐: 截断到 {min_length} 帧, 数据损失 {data_loss:.1%}")
        
        # 截断并提取数据
        aligned_data = {}
        for topic, messages in topic_messages.items():
            truncated_messages = messages[:min_length]
            aligned_data[topic] = [msg['data'] for msg in truncated_messages]
            
            lost = len(messages) - min_length
            if lost > 0:
                self._log_warning(f"Topic {topic}: 丢失 {lost} 条消息")
        
        return aligned_data


class InterpolationTimeAligner(TimeAligner):
    """插值时间对齐器 - 基于时间插值对齐"""
    
    def align_topics(self, topic_messages: Dict[str, List[Dict]]) -> Dict[str, List[Any]]:
        """使用插值对齐到基准时间线"""
        if not topic_messages:
            return {}
        
        # 选择基准topic
        base_topic = self._select_base_topic(topic_messages)
        base_messages = topic_messages[base_topic]
        base_timestamps = [msg['timestamp'] for msg in base_messages]
        target_length = len(base_messages)
        
        self._log_info(f"插值对齐: 基准topic '{base_topic}', 目标长度 {target_length}")
        
        # 对齐所有topic
        aligned_data = {}
        alignment_quality = {}
        
        for topic, messages in topic_messages.items():
            if topic == base_topic:
                # 基准topic直接提取数据
                aligned_data[topic] = [msg['data'] for msg in messages]
                alignment_quality[topic] = 1.0
            else:
                # 其他topic进行时间对齐
                aligned_messages, quality = self._align_topic_to_timeline(
                    messages, base_timestamps, topic
                )
                aligned_data[topic] = aligned_messages
                alignment_quality[topic] = quality
        
        # 报告对齐质量
        avg_quality = np.mean(list(alignment_quality.values()))
        self._log_info(f"平均对齐质量: {avg_quality:.3f}")
        
        # 检查低质量对齐
        for topic, quality in alignment_quality.items():
            if quality < self.config.min_sync_quality:
                self._log_warning(f"Topic {topic} 对齐质量低: {quality:.3f}")
        
        return aligned_data
    
    def _align_topic_to_timeline(self, messages: List[Dict], target_timestamps: List[int], topic_name: str) -> Tuple[List[Any], float]:
        """将单个topic对齐到目标时间线"""
        if not messages:
            return [], 0.0
        
        source_timestamps = np.array([msg['timestamp'] for msg in messages])
        source_data = [msg['data'] for msg in messages]
        target_ts = np.array(target_timestamps)
        
        aligned_data = []
        time_diffs = []
        
        for target_t in target_ts:
            # 找到最接近的时间戳
            time_diff_array = np.abs(source_timestamps - target_t)
            nearest_idx = np.argmin(time_diff_array)
            time_diff_ns = time_diff_array[nearest_idx]
            time_diff_ms = time_diff_ns / 1e6
            
            time_diffs.append(time_diff_ms)
            
            # 检查时间差
            if time_diff_ms > self.config.max_time_diff_ms:
                self._log_warning(f"{topic_name}: 时间差过大 {time_diff_ms:.1f}ms")
            elif time_diff_ms > self.config.warn_time_diff_ms:
                self._log_warning(f"{topic_name}: 时间差较大 {time_diff_ms:.1f}ms")
            
            aligned_data.append(source_data[nearest_idx])
        
        # 计算对齐质量
        if time_diffs:
            avg_time_diff = np.mean(time_diffs)
            # 质量分数基于时间差（越小越好）
            quality = max(0.0, 1.0 - (avg_time_diff / self.config.max_time_diff_ms))
        else:
            quality = 0.0
        
        return aligned_data, quality


class NearestNeighborTimeAligner(TimeAligner):
    """最近邻时间对齐器 - 选择时间上最近的消息"""
    
    def align_topics(self, topic_messages: Dict[str, List[Dict]]) -> Dict[str, List[Any]]:
        """使用最近邻策略对齐"""
        # 实现与插值对齐类似，但使用更严格的时间匹配
        return InterpolationTimeAligner(self.config, self.logger).align_topics(topic_messages)


def create_time_aligner(config: AlignmentConfig, logger=None) -> TimeAligner:
    """工厂函数：根据配置创建时间对齐器"""
    
    if config.strategy == AlignmentStrategy.EMERGENCY:
        return EmergencyTimeAligner(config, logger)
    elif config.strategy == AlignmentStrategy.INTERPOLATION:
        return InterpolationTimeAligner(config, logger)
    elif config.strategy == AlignmentStrategy.NEAREST_NEIGHBOR:
        return NearestNeighborTimeAligner(config, logger)
    else:
        raise ValueError(f"Unknown alignment strategy: {config.strategy}")


# 便利函数
def emergency_align_topics(topic_messages: Dict[str, List[Dict]], logger=None) -> Dict[str, List[Any]]:
    """快速应急对齐"""
    config = AlignmentConfig.emergency_config()
    aligner = EmergencyTimeAligner(config, logger)
    return aligner.align_topics(topic_messages)


def production_align_topics(topic_messages: Dict[str, List[Dict]], logger=None) -> Dict[str, List[Any]]:
    """生产环境对齐"""
    config = AlignmentConfig.production_config()
    aligner = InterpolationTimeAligner(config, logger)
    return aligner.align_topics(topic_messages)


def galaxea_align_topics(topic_messages: Dict[str, List[Dict]], logger=None) -> Dict[str, List[Any]]:
    """Galaxea机器人优化对齐"""
    config = AlignmentConfig.galaxea_config()
    aligner = InterpolationTimeAligner(config, logger)
    return aligner.align_topics(topic_messages)
