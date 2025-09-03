#!/usr/bin/env python3
"""
基于工作代码的Bag文件Topic检查器
"""

from pathlib import Path
from rosbags.rosbag1 import Reader

def inspect_bag_file(bag_file: Path):
    """检查单个bag文件的topics"""
    print(f"🔍 检查文件: {bag_file.name}")
    
    try:
        with Reader(bag_file) as reader:
            print("📋 发现的Topics:")
            for connection in reader.connections:
                print(f"  • {connection.topic}")
                print(f"    类型: {connection.msgtype}")
                print(f"    消息数: {connection.msgcount}")
                print()
                
    except Exception as e:
        print(f"❌ 解析失败: {e}")

def main():
    dataset_path = Path("/home/diy01/dev/dataset_example")
    
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return
    
    print("🚀 Bag文件Topic检查器")
    print("=" * 40)
    
    # 查找bag文件
    bag_files = list(dataset_path.glob("*.bag"))
    print(f"找到 {len(bag_files)} 个bag文件")
    print()
    
    # 只检查第一个文件，避免输出过多
    if bag_files:
        inspect_bag_file(bag_files[0])
        
        # 收集所有唯一的topics
        all_topics = set()
        all_topic_info = {}
        
        for bag_file in bag_files:
            try:
                with Reader(bag_file) as reader:
                    for connection in reader.connections:
                        all_topics.add(connection.topic)
                        if connection.topic not in all_topic_info:
                            all_topic_info[connection.topic] = {
                                'msgtype': connection.msgtype,
                                'total_messages': connection.msgcount
                            }
                        else:
                            all_topic_info[connection.topic]['total_messages'] += connection.msgcount
            except Exception as e:
                print(f"⚠️ 跳过文件 {bag_file.name}: {e}")
        
        print("🌟 所有Topics汇总:")
        for topic in sorted(all_topics):
            info = all_topic_info[topic]
            print(f"  • {topic}")
            print(f"    类型: {info['msgtype']}")
            print(f"    总消息数: {info['total_messages']}")
            print()
        
        # 生成配置建议
        print("🛠️ 配置建议:")
        generate_config_suggestions(all_topic_info)

def generate_config_suggestions(topic_info):
    """基于topics生成配置建议"""
    
    print("# 基于发现的topics的配置建议")
    print()
    
    image_topics = []
    state_topics = []
    
    for topic, info in topic_info.items():
        msgtype = info['msgtype']
        
        # 图像topics
        if 'Image' in msgtype or 'image' in topic.lower():
            image_topics.append((topic, msgtype))
        
        # 状态topics
        elif 'joint' in msgtype.lower() or 'joint' in topic.lower() or 'state' in topic.lower():
            state_topics.append((topic, msgtype))
    
    print("## 图像配置 (images):")
    for i, (topic, msgtype) in enumerate(image_topics):
        cam_name = f"cam_{i+1}"
        if 'left' in topic.lower():
            cam_name += "_left"
        elif 'right' in topic.lower():
            cam_name += "_right"
        elif 'head' in topic.lower():
            cam_name += "_head"
            
        print(f"""
    - cam_name: {cam_name}
      args:
        topic_name: "{topic}"
        # 消息类型: {msgtype}""")
    
    print("\n## 状态配置 (state):")
    for topic, msgtype in state_topics:
        print(f"""
    - names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]  # 根据实际调整
      args:
        topic_name: "{topic}"
        range_from: 0
        range_to: 7
        # 消息类型: {msgtype}""")

if __name__ == "__main__":
    main()
