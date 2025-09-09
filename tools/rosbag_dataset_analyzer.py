#!/usr/bin/env python3
"""
ROS Bag 数据集结构探测脚本

用于分析实际的rosbag数据集结构，包括：
1. bag文件中的topic列表
2. 各topic的消息类型和数量
3. JSON元数据文件的内容
4. 生成适合的配置文件模板
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

try:
    from rosbags.highlevel import AnyReader
    HAS_ROSBAGS = True
except ImportError:
    print("⚠️  rosbags库未安装，无法解析bag文件内容")
    HAS_ROSBAGS = False


def analyze_dataset_structure(dataset_path: Path) -> Dict:
    """
    分析数据集目录结构
    
    Args:
        dataset_path: 数据集根目录
        
    Returns:
        dict: 包含分析结果的字典
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    analysis = {
        'dataset_path': str(dataset_path),
        'bag_files': [],
        'json_files': [],
        'topics_analysis': {},
        'metadata_analysis': {},
        'recommended_config': {}
    }
    
    # 查找bag文件
    bag_files = []
    for pattern in ['*.bag', '*.mcap', '*.db3']:
        bag_files.extend(list(dataset_path.rglob(pattern)))
    
    analysis['bag_files'] = [str(f) for f in bag_files]
    
    # 查找JSON文件
    json_files = list(dataset_path.rglob('*.json'))
    analysis['json_files'] = [str(f) for f in json_files]
    
    # 分析bag文件内容
    if HAS_ROSBAGS and bag_files:
        print(f"🔍 分析 {len(bag_files)} 个bag文件...")
        analysis['topics_analysis'] = analyze_bag_topics(bag_files[:3])  # 只分析前3个文件
    
    # 分析JSON元数据
    if json_files:
        print(f"📋 分析 {len(json_files)} 个JSON文件...")
        analysis['metadata_analysis'] = analyze_json_metadata(json_files[:5])  # 只分析前5个文件
    
    # 生成推荐配置
    analysis['recommended_config'] = generate_recommended_config(analysis)
    
    return analysis


def analyze_bag_topics(bag_files: List[Path]) -> Dict:
    """分析bag文件中的topics"""
    topics_info = {}
    
    for bag_file in bag_files:
        print(f"  📁 分析文件: {bag_file.name}")
        
        try:
            with AnyReader([bag_file]) as reader:
                # 获取连接信息
                for connection in reader.connections.values():
                    topic = connection.topic
                    if topic not in topics_info:
                        topics_info[topic] = {
                            'msgtype': connection.msgtype,
                            'message_counts': [],
                            'files': []
                        }
                    
                    # 统计消息数量
                    msg_count = 0
                    for conn, timestamp, rawdata in reader.messages():
                        if conn.topic == topic:
                            msg_count += 1
                    
                    topics_info[topic]['message_counts'].append(msg_count)
                    topics_info[topic]['files'].append(str(bag_file))
                    
        except Exception as e:
            print(f"    ❌ 解析失败: {e}")
            continue
    
    # 汇总统计
    for topic, info in topics_info.items():
        info['total_messages'] = sum(info['message_counts'])
        info['avg_messages_per_file'] = info['total_messages'] / len(info['message_counts']) if info['message_counts'] else 0
        info['files_count'] = len(info['files'])
    
    return topics_info


def analyze_json_metadata(json_files: List[Path]) -> Dict:
    """分析JSON元数据文件"""
    metadata_samples = []
    common_keys = set()
    
    for json_file in json_files:
        print(f"  📄 分析文件: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata_samples.append({
                    'file': str(json_file),
                    'keys': list(data.keys()),
                    'sample_data': {k: str(v)[:100] + ('...' if len(str(v)) > 100 else '') 
                                   for k, v in data.items()}
                })
                
                if not common_keys:
                    common_keys = set(data.keys())
                else:
                    common_keys &= set(data.keys())
                    
        except Exception as e:
            print(f"    ❌ 解析失败: {e}")
            continue
    
    return {
        'samples': metadata_samples[:3],  # 只保留前3个样本
        'common_keys': list(common_keys),
        'total_files': len(json_files)
    }


def generate_recommended_config(analysis: Dict) -> Dict:
    """基于分析结果生成推荐配置"""
    config = {
        'fps': 30,  # 默认值，需要根据实际调整
        'features': {
            'observation': {
                'images': [],
                'state': {
                    'sub_state': []
                }
            },
            'action': {
                'sub_action': []
            }
        }
    }
    
    # 根据topics分析生成图像配置
    topics = analysis.get('topics_analysis', {})
    cam_counter = 1
    
    for topic, info in topics.items():
        msgtype = info.get('msgtype', '')
        
        # 图像topics
        if 'Image' in msgtype or 'image' in topic.lower():
            cam_name = f"cam_{cam_counter}"
            if 'left' in topic.lower():
                cam_name += "_left"
            elif 'right' in topic.lower():
                cam_name += "_right"
            elif 'head' in topic.lower():
                cam_name += "_head"
            
            config['features']['observation']['images'].append({
                'cam_name': cam_name,
                'args': {
                    'topic_name': topic
                }
            })
            cam_counter += 1
        
        # 状态topics（关节状态等）
        elif 'JointState' in msgtype or 'joint' in topic.lower():
            config['features']['observation']['state']['sub_state'].append({
                'names': [f'joint_{i}' for i in range(1, 8)],  # 假设7个关节
                'args': {
                    'topic_name': topic,
                    'range_from': 0,
                    'range_to': 7
                }
            })
    
    return config


def generate_report(analysis: Dict, output_path: Path = None):
    """生成分析报告"""
    report = []
    report.append("# ROS Bag 数据集结构分析报告")
    report.append("=" * 50)
    report.append("")
    
    report.append(f"📁 数据集路径: {analysis['dataset_path']}")
    report.append(f"📦 Bag文件数量: {len(analysis['bag_files'])}")
    report.append(f"📄 JSON文件数量: {len(analysis['json_files'])}")
    report.append("")
    
    # Topics分析
    if analysis['topics_analysis']:
        report.append("🎯 Topics 分析:")
        for topic, info in analysis['topics_analysis'].items():
            report.append(f"  • {topic}")
            report.append(f"    类型: {info['msgtype']}")
            report.append(f"    总消息数: {info['total_messages']}")
            report.append(f"    平均消息数/文件: {info['avg_messages_per_file']:.1f}")
            report.append("")
    
    # 元数据分析
    if analysis['metadata_analysis']:
        report.append("📋 JSON 元数据分析:")
        common_keys = analysis['metadata_analysis'].get('common_keys', [])
        if common_keys:
            report.append(f"  通用字段: {', '.join(common_keys)}")
            report.append("")
            
            # 显示样本数据
            for sample in analysis['metadata_analysis']['samples']:
                report.append(f"  📄 {Path(sample['file']).name}:")
                for key, value in sample['sample_data'].items():
                    report.append(f"    {key}: {value}")
                report.append("")
    
    # 推荐配置
    report.append("🛠️  推荐配置 (convertor_config.yaml):")
    report.append("```yaml")
    config_yaml = yaml.dump(analysis['recommended_config'], default_flow_style=False, indent=2)
    report.append(config_yaml)
    report.append("```")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"📝 报告已保存到: {output_path}")
    else:
        print(report_text)


def main():
    """主函数"""
    print("🚀 ROS Bag 数据集结构探测工具")
    print("=" * 50)
    
    # 这里需要指定实际的数据集路径
    dataset_path = Path("/home/diy01/dev/dataset_example")
    
    if not dataset_path.exists():
        print("❌ 请修改 dataset_path 为实际的数据集路径")
        print("💡 使用方法:")
        print("   1. 修改脚本中的 dataset_path 变量")
        print("   2. 确保安装了 rosbags 库: pip install rosbags")
        print("   3. 重新运行脚本")
        return
    
    try:
        # 执行分析
        analysis = analyze_dataset_structure(dataset_path)
        
        # 生成报告
        report_path = Path("rosbag_dataset_analysis_report.md")
        generate_report(analysis, report_path)
        
        print("\n🎉 分析完成！")
        print("💡 接下来的步骤:")
        print("   1. 查看生成的分析报告")
        print("   2. 根据推荐配置调整 convertor_config_*.yaml")
        print("   3. 测试转换流程")
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
