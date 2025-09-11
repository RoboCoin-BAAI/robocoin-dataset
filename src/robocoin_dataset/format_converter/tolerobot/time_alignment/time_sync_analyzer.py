"""
时间同步分析器
用于分析RosBag数据的时间分布和同步质量
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TopicTimeStats:
    """单个topic的时间统计信息"""
    topic_name: str
    message_count: int
    time_range_sec: float
    avg_frequency_hz: float
    std_frequency_hz: float
    time_gaps_ms: list[float]
    max_gap_ms: float
    min_gap_ms: float


@dataclass
class SyncQualityReport:
    """时间同步质量报告"""
    overall_quality_score: float  # 0-1之间，1为完美同步
    topic_stats: dict[str, TopicTimeStats]
    alignment_feasibility: str  # "excellent", "good", "poor", "impossible"
    recommended_strategy: str
    base_topic_candidates: list[str]
    potential_issues: list[str]
    data_loss_estimate: float  # 预估的数据损失百分比


class TimeSyncAnalyzer:
    """时间同步分析器"""
    
    def __init__(self, logger=None) -> None:  # noqa: ANN001
        self.logger = logger
        
    def analyze_topic_messages(self, topic_messages: dict[str, list[dict]]) -> SyncQualityReport:
        """
        分析topic消息的时间同步质量
        
        Args:
            topic_messages: {topic_name: [{'timestamp': xxx, 'data': xxx}, ...]}
            
        Returns:
            SyncQualityReport: 同步质量报告
        """
        topic_stats = {}
        
        # 分析每个topic的时间统计
        for topic_name, messages in topic_messages.items():
            if not messages:
                continue
                
            topic_stats[topic_name] = self._analyze_topic_timing(topic_name, messages)
        
        # 计算整体同步质量
        quality_score = self._calculate_sync_quality(topic_stats)
        
        # 分析对齐可行性
        feasibility = self._assess_alignment_feasibility(topic_stats)
        
        # 推荐策略
        strategy = self._recommend_strategy(topic_stats, quality_score)
        
        # 选择基准topic候选
        base_candidates = self._select_base_topic_candidates(topic_stats)
        
        # 识别潜在问题
        issues = self._identify_issues(topic_stats)
        
        # 估算数据损失
        data_loss = self._estimate_data_loss(topic_stats)
        
        return SyncQualityReport(
            overall_quality_score=quality_score,
            topic_stats=topic_stats,
            alignment_feasibility=feasibility,
            recommended_strategy=strategy,
            base_topic_candidates=base_candidates,
            potential_issues=issues,
            data_loss_estimate=data_loss
        )
    
    def _analyze_topic_timing(self, topic_name: str, messages: list[dict]) -> TopicTimeStats:
        """分析单个topic的时间统计"""
        timestamps = [msg['timestamp'] for msg in messages]
        
        if len(timestamps) < 2:
            return TopicTimeStats(
                topic_name=topic_name,
                message_count=len(timestamps),
                time_range_sec=0.0,
                avg_frequency_hz=0.0,
                std_frequency_hz=0.0,
                time_gaps_ms=[],
                max_gap_ms=0.0,
                min_gap_ms=0.0
            )
        
        # 转换为秒
        timestamps_sec = [ts / 1e9 for ts in timestamps]
        time_range = max(timestamps_sec) - min(timestamps_sec)
        
        # 计算时间间隔
        time_gaps = [timestamps_sec[i+1] - timestamps_sec[i] for i in range(len(timestamps_sec)-1)]
        time_gaps_ms = [gap * 1000 for gap in time_gaps]
        
        # 频率统计
        if time_range > 0:
            avg_frequency = len(timestamps) / time_range
        else:
            avg_frequency = 0.0
        
        # 瞬时频率标准差
        if len(time_gaps) > 0:
            instant_freqs = [1.0 / gap for gap in time_gaps if gap > 0]
            std_frequency = np.std(instant_freqs) if instant_freqs else 0.0
        else:
            std_frequency = 0.0
        
        return TopicTimeStats(
            topic_name=topic_name,
            message_count=len(timestamps),
            time_range_sec=time_range,
            avg_frequency_hz=avg_frequency,
            std_frequency_hz=std_frequency,
            time_gaps_ms=time_gaps_ms,
            max_gap_ms=max(time_gaps_ms) if time_gaps_ms else 0.0,
            min_gap_ms=min(time_gaps_ms) if time_gaps_ms else 0.0
        )
    
    def _calculate_sync_quality(self, topic_stats: dict[str, TopicTimeStats]) -> float:
        """计算整体同步质量分数 (0-1)"""
        if not topic_stats:
            return 0.0
        
        scores = []
        
        for stats in topic_stats.values():
            # 频率稳定性得分
            if stats.avg_frequency_hz > 0:
                freq_stability = max(0, 1 - (stats.std_frequency_hz / stats.avg_frequency_hz))
            else:
                freq_stability = 0.0
            
            # 时间间隔均匀性得分
            if stats.time_gaps_ms:
                gap_uniformity = max(0, 1 - (np.std(stats.time_gaps_ms) / np.mean(stats.time_gaps_ms)))
            else:
                gap_uniformity = 0.0
            
            # 综合得分
            topic_score = (freq_stability + gap_uniformity) / 2
            scores.append(topic_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _assess_alignment_feasibility(self, topic_stats: dict[str, TopicTimeStats]) -> str:
        """评估对齐可行性"""
        if not topic_stats:
            return "impossible"
        
        frequencies = [stats.avg_frequency_hz for stats in topic_stats.values() if stats.avg_frequency_hz > 0]
        
        if not frequencies:
            return "impossible"
        
        freq_ratio = max(frequencies) / min(frequencies)
        
        if freq_ratio <= 2:
            return "excellent"
        if freq_ratio <= 5:
            return "good" 
        if freq_ratio <= 20:
            return "poor"
        return "impossible"
    
    def _recommend_strategy(self, topic_stats: dict[str, TopicTimeStats], quality_score: float) -> str:
        """推荐对齐策略"""
        if quality_score > 0.8:
            return "interpolation"
        if quality_score > 0.5:
            return "nearest_neighbor"
        return "emergency"
    
    def _select_base_topic_candidates(self, topic_stats: dict[str, TopicTimeStats]) -> list[str]:
        """选择基准topic候选"""
        
        # 优先级关键词
        priority_keywords = ['rgb', 'color', 'image', 'camera', 'compressed', 'depth']
        
        # 按优先级和频率稳定性排序
        topic_scores = []
        for topic_name, stats in topic_stats.items():
            # 关键词匹配得分
            keyword_score = 0
            for i, keyword in enumerate(priority_keywords):
                if keyword in topic_name.lower():
                    keyword_score = len(priority_keywords) - i
                    break
            
            # 频率稳定性得分
            if stats.avg_frequency_hz > 0 and stats.std_frequency_hz >= 0:
                stability_score = max(0, 1 - (stats.std_frequency_hz / stats.avg_frequency_hz))
            else:
                stability_score = 0
            
            # 消息数量得分（归一化）
            msg_count_score = min(1.0, stats.message_count / 1000)
            
            total_score = keyword_score * 0.5 + stability_score * 0.3 + msg_count_score * 0.2
            topic_scores.append((topic_name, total_score))
        
        # 排序并返回前三名
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in topic_scores[:3]]
    
    def _identify_issues(self, topic_stats: dict[str, TopicTimeStats]) -> list[str]:
        """识别潜在问题"""
        issues = []
        
        if not topic_stats:
            issues.append("No topic data available")
            return issues
        
        # 检查频率差异
        frequencies = [stats.avg_frequency_hz for stats in topic_stats.values() if stats.avg_frequency_hz > 0]
        if frequencies:
            freq_ratio = max(frequencies) / min(frequencies)
            if freq_ratio > 10:
                issues.append(f"Extreme frequency differences (ratio: {freq_ratio:.1f}x)")
            elif freq_ratio > 5:
                issues.append(f"High frequency differences (ratio: {freq_ratio:.1f}x)")
        
        # 检查频率不稳定
        unstable_topics = [
            stats.topic_name for stats in topic_stats.values()
            if stats.avg_frequency_hz > 0 and (stats.std_frequency_hz / stats.avg_frequency_hz) > 0.5
        ]
        if unstable_topics:
            issues.append(f"Unstable frequencies in topics: {', '.join(unstable_topics[:3])}")
        
        # 检查数据稀少
        sparse_topics = [
            stats.topic_name for stats in topic_stats.values()
            if stats.message_count < 10
        ]
        if sparse_topics:
            issues.append(f"Sparse data in topics: {', '.join(sparse_topics[:3])}")
        
        # 检查时间间隔异常
        gap_issues = []
        for stats in topic_stats.values():
            if stats.time_gaps_ms and stats.max_gap_ms > 1000:  # 1秒以上的间隔
                gap_issues.append(stats.topic_name)  # noqa: PERF401
        if gap_issues:
            issues.append(f"Large time gaps in topics: {', '.join(gap_issues[:3])}")
        
        return issues
    
    def _estimate_data_loss(self, topic_stats: dict[str, TopicTimeStats]) -> float:
        """估算使用应急策略的数据损失百分比"""
        if not topic_stats:
            return 0.0
        
        message_counts = [stats.message_count for stats in topic_stats.values()]
        min_count = min(message_counts)
        total_messages = sum(message_counts)
        
        if total_messages == 0:
            return 0.0
        
        preserved_messages = min_count * len(message_counts)
        return (total_messages - preserved_messages) / total_messages
        
    
    def print_analysis_report(self, report: SyncQualityReport) -> None:
        """打印分析报告"""
        print("=" * 80)
        print("🔍 ROS Bag 时间同步质量分析报告")
        print("=" * 80)
        
        print("\n📊 整体评估:")
        print(f"   同步质量得分: {report.overall_quality_score:.3f} / 1.000")
        print(f"   对齐可行性: {report.alignment_feasibility}")
        print(f"   推荐策略: {report.recommended_strategy}")
        print(f"   预估数据损失: {report.data_loss_estimate:.1%}")
        
        print("\n🎯 推荐基准Topics:")
        for i, topic in enumerate(report.base_topic_candidates, 1):
            print(f"   {i}. {topic}")
        
        print("\n📈 Topic 统计信息:")
        for topic, stats in report.topic_stats.items():
            print(f"\n   📌 {topic}")
            print(f"      消息数量: {stats.message_count}")
            print(f"      平均频率: {stats.avg_frequency_hz:.1f} Hz")
            print(f"      频率稳定性: ±{stats.std_frequency_hz:.1f} Hz")
            print(f"      时间跨度: {stats.time_range_sec:.1f} 秒")
            if stats.time_gaps_ms:
                print(f"      时间间隔: {stats.min_gap_ms:.1f} - {stats.max_gap_ms:.1f} ms")
        
        if report.potential_issues:
            print("\n⚠️  潜在问题:")
            for issue in report.potential_issues:
                print(f"   • {issue}")
        
        print("\n💡 建议:")
        if report.alignment_feasibility == "excellent":
            print("   • 时间同步质量优秀，可以使用插值对齐获得最佳效果")
        elif report.alignment_feasibility == "good":
            print("   • 时间同步质量良好，建议使用插值或最近邻对齐")
        elif report.alignment_feasibility == "poor":
            print("   • 时间同步质量较差，建议使用最近邻或应急对齐")
        else:
            print("   • 时间同步困难，建议使用应急对齐或考虑数据预处理")
        
        print("=" * 80)
