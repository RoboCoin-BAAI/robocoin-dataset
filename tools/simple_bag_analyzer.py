#!/usr/bin/env python3
"""
简化版的 ROS Bag 分析脚本，专门用于修复bag文件解析问题
"""

import json
from pathlib import Path
import yaml

try:
    from rosbags.rosbag1 import Reader as Rosbag1Reader
    from rosbags.rosbag2 import Reader as Rosbag2Reader  
    HAS_ROSBAGS = True
except ImportError:
    try:
        from rosbags.highlevel import AnyReader
        HAS_ROSBAGS = True
        USE_HIGHLEVEL = True
    except ImportError:
        print("⚠️  rosbags库未安装")
        HAS_ROSBAGS = False

def analyze_single_bag_simple(bag_file: Path) -> dict:
    """使用简化方法分析单个bag文件"""
    print(f"  📁 分析文件: {bag_file.name}")
    
    topics_info = {}
    
    try:
        # 尝试使用不同的Reader
        if bag_file.suffix == '.bag':
            # ROS1 bag files
            with Rosbag1Reader(bag_file) as reader:
                connections = reader.connections
                for conn_id, connection in connections.items():
                    topic = connection.topic
                    topics_info[topic] = {
                        'msgtype': connection.msgtype,
                        'msgcount': connection.msgcount,
                        'file': str(bag_file)
                    }
        else:
            print(f"    ❓ 未知格式: {bag_file.suffix}")
            
    except Exception as e:
        print(f"    ❌ 解析失败: {e}")
        return {}
    
    return topics_info

def main():
    dataset_path = Path("/home/diy01/dev/dataset_example")
    
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    print("🔍 简化版Bag文件分析")
    print("=" * 40)
    
    # 查找bag文件
    bag_files = list(dataset_path.rglob("*.bag"))
    print(f"找到 {len(bag_files)} 个bag文件")
    
    all_topics = {}
    
    for bag_file in bag_files[:2]:  # 只分析前2个
        topics = analyze_single_bag_simple(bag_file)
        for topic, info in topics.items():
            if topic not in all_topics:
                all_topics[topic] = info
            else:
                # 合并信息
                all_topics[topic]['msgcount'] += info['msgcount']
    
    print("\n📋 发现的Topics:")
    for topic, info in all_topics.items():
        print(f"  • {topic}")
        print(f"    类型: {info['msgtype']}")
        print(f"    消息数: {info['msgcount']}")
        print("")
    
    # 生成配置建议
    print("🛠️ 根据发现的topics生成配置建议:")
    
    config = {
        'fps': 30,
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
    
    cam_counter = 1
    
    for topic, info in all_topics.items():
        msgtype = info['msgtype']
        
        # 图像相关的topics
        if any(keyword in msgtype.lower() for keyword in ['image', 'compressed']):
            cam_name = f"cam_{cam_counter}"
            
            # 根据topic名称推断相机位置
            if 'left' in topic.lower():
                cam_name += '_left'
            elif 'right' in topic.lower():
                cam_name += '_right'
            elif 'head' in topic.lower():
                cam_name += '_head'
            
            config['features']['observation']['images'].append({
                'cam_name': cam_name,
                'args': {
                    'topic_name': topic
                },
                'comment': f'从 {msgtype} 推断的图像topic'
            })
            cam_counter += 1
        
        # 关节状态相关
        elif 'joint' in msgtype.lower() or 'joint' in topic.lower():
            config['features']['observation']['state']['sub_state'].append({
                'names': ['joint_' + str(i) for i in range(1, 8)],  # 假设7个关节
                'args': {
                    'topic_name': topic,
                    'range_from': 0,
                    'range_to': 7
                },
                'comment': f'从 {msgtype} 推断的关节状态'
            })
    
    print("\n建议的配置文件内容:")
    print("```yaml")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    print("```")

if __name__ == "__main__":
    main()
