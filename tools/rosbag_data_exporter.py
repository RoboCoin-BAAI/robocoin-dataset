#!/usr/bin/env python3
"""
ROS Bag数据导出工具
用于将rosbag中的数据展开到不同格式的文件中
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, register_types
import argparse
import yaml
from datetime import datetime
import pandas as pd


class RosbagDataExporter:
    def __init__(self, bag_path: str, output_dir: str):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.images_dir = self.output_dir / "images"
        self.data_dir = self.output_dir / "data" 
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.data_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.topic_data = {}
        self.topic_info = {}
        
    def export_all_data(self):
        """导出所有数据"""
        print(f"🚀 开始导出rosbag数据: {self.bag_path}")
        
        with AnyReader([self.bag_path]) as reader:
            # 获取topic信息
            connections = reader.connections
            topics = reader.topics
            
            self.topic_info = {}
            for connection in connections:
                topic = connection.topic
                self.topic_info[topic] = {
                    'msg_type': str(connection.msgtype),
                    'msg_count': connection.msgcount,
                    'frequency': connection.msgcount / reader.duration * 1e9 if reader.duration > 0 else 0
                }
            
            print(f"📊 发现 {len(self.topic_info)} 个topics")
            print(f"⏱️ 总时长: {reader.duration / 1e9:.2f} 秒")
            
            # 初始化topic数据存储
            for topic in self.topic_info.keys():
                self.topic_data[topic] = []
            
            # 读取所有消息
            message_count = 0
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # 存储消息数据
                self.topic_data[topic].append({
                    'timestamp': timestamp,
                    'data': msg
                })
                
                message_count += 1
                if message_count % 1000 == 0:
                    print(f"📝 已处理 {message_count} 条消息...")
        
        # 导出不同类型的数据
        self._export_topic_info()
        self._export_image_data()
        self._export_sensor_data()
        self._export_control_data()
        self._export_json_metadata()
        
        print(f"✅ 数据导出完成! 输出目录: {self.output_dir}")
    
    def _export_topic_info(self):
        """导出topic信息"""
        topic_info_file = self.metadata_dir / "topic_info.yaml"
        with open(topic_info_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.topic_info, f, allow_unicode=True, default_flow_style=False)
        
        print(f"📋 Topic信息已导出到: {topic_info_file}")
    
    def _export_image_data(self):
        """导出图像数据"""
        image_topics = [topic for topic in self.topic_data.keys() 
                       if 'camera' in topic or 'image' in topic]
        
        if not image_topics:
            print("📷 未发现图像topic")
            return
        
        print(f"📷 发现 {len(image_topics)} 个图像topics")
        
        for topic in image_topics:
            topic_name = topic.replace('/', '_').strip('_')
            topic_dir = self.images_dir / topic_name
            topic_dir.mkdir(exist_ok=True)
            
            print(f"  处理图像topic: {topic}")
            
            for i, msg_data in enumerate(self.topic_data[topic]):
                try:
                    msg = msg_data['data']
                    timestamp = msg_data['timestamp']
                    
                    if hasattr(msg, 'data'):  # CompressedImage
                        # 解码压缩图像
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if image is not None:
                            filename = f"{timestamp}_{i:06d}.jpg"
                            cv2.imwrite(str(topic_dir / filename), image)
                    
                    elif hasattr(msg, 'width') and hasattr(msg, 'height'):  # Raw Image
                        # 处理原始图像数据
                        if msg.encoding == 'rgb8':
                            image = np.frombuffer(msg.data, dtype=np.uint8)
                            image = image.reshape((msg.height, msg.width, 3))
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        elif msg.encoding == 'bgr8':
                            image = np.frombuffer(msg.data, dtype=np.uint8)
                            image = image.reshape((msg.height, msg.width, 3))
                        elif msg.encoding == 'mono8':
                            image = np.frombuffer(msg.data, dtype=np.uint8)
                            image = image.reshape((msg.height, msg.width))
                        elif msg.encoding == '16UC1':  # 深度图像
                            image = np.frombuffer(msg.data, dtype=np.uint16)
                            image = image.reshape((msg.height, msg.width))
                            # 转换为8位用于显示
                            image = (image / 256).astype(np.uint8)
                        else:
                            print(f"    未支持的图像编码: {msg.encoding}")
                            continue
                        
                        filename = f"{timestamp}_{i:06d}.png"
                        cv2.imwrite(str(topic_dir / filename), image)
                    
                except Exception as e:
                    print(f"    处理图像失败 {i}: {e}")
            
            print(f"    ✅ {topic} - 导出了 {len(self.topic_data[topic])} 张图像")
    
    def _export_sensor_data(self):
        """导出传感器数据"""
        sensor_topics = [topic for topic in self.topic_data.keys() 
                        if any(keyword in topic.lower() for keyword in 
                              ['imu', 'feedback', 'joint', 'state', 'pose'])]
        
        if not sensor_topics:
            print("🔧 未发现传感器topic")
            return
        
        print(f"🔧 发现 {len(sensor_topics)} 个传感器topics")
        
        for topic in sensor_topics:
            topic_name = topic.replace('/', '_').strip('_')
            
            # 准备数据列表
            data_rows = []
            
            for msg_data in self.topic_data[topic]:
                msg = msg_data['data']
                timestamp = msg_data['timestamp']
                
                row = {'timestamp': timestamp}
                
                try:
                    if hasattr(msg, 'position') and hasattr(msg, 'velocity'):  # JointState
                        for i, name in enumerate(msg.name if hasattr(msg, 'name') else []):
                            if i < len(msg.position):
                                row[f'{name}_position'] = msg.position[i]
                            if i < len(msg.velocity):
                                row[f'{name}_velocity'] = msg.velocity[i]
                            if hasattr(msg, 'effort') and i < len(msg.effort):
                                row[f'{name}_effort'] = msg.effort[i]
                    
                    elif hasattr(msg, 'linear_acceleration'):  # IMU
                        row['accel_x'] = msg.linear_acceleration.x
                        row['accel_y'] = msg.linear_acceleration.y
                        row['accel_z'] = msg.linear_acceleration.z
                        row['gyro_x'] = msg.angular_velocity.x
                        row['gyro_y'] = msg.angular_velocity.y
                        row['gyro_z'] = msg.angular_velocity.z
                        
                        if hasattr(msg, 'orientation'):
                            row['orient_x'] = msg.orientation.x
                            row['orient_y'] = msg.orientation.y
                            row['orient_z'] = msg.orientation.z
                            row['orient_w'] = msg.orientation.w
                    
                    elif hasattr(msg, 'pose'):  # PoseStamped
                        row['pos_x'] = msg.pose.position.x
                        row['pos_y'] = msg.pose.position.y
                        row['pos_z'] = msg.pose.position.z
                        row['orient_x'] = msg.pose.orientation.x
                        row['orient_y'] = msg.pose.orientation.y
                        row['orient_z'] = msg.pose.orientation.z
                        row['orient_w'] = msg.pose.orientation.w
                    
                    elif hasattr(msg, 'twist'):  # TwistStamped
                        twist = msg.twist if hasattr(msg, 'twist') else msg
                        row['linear_x'] = twist.linear.x
                        row['linear_y'] = twist.linear.y
                        row['linear_z'] = twist.linear.z
                        row['angular_x'] = twist.angular.x
                        row['angular_y'] = twist.angular.y
                        row['angular_z'] = twist.angular.z
                    
                    else:
                        # 通用处理：尝试提取所有数值字段
                        row.update(self._extract_numeric_fields(msg, ''))
                
                except Exception as e:
                    print(f"    处理传感器数据失败: {e}")
                    continue
                
                data_rows.append(row)
            
            # 保存为CSV
            if data_rows:
                df = pd.DataFrame(data_rows)
                csv_file = self.data_dir / f"{topic_name}_sensor.csv"
                df.to_csv(csv_file, index=False)
                print(f"  ✅ {topic} - 导出了 {len(data_rows)} 行传感器数据到 {csv_file}")
    
    def _export_control_data(self):
        """导出控制数据"""
        control_topics = [topic for topic in self.topic_data.keys() 
                         if any(keyword in topic.lower() for keyword in 
                               ['control', 'target', 'command', 'cmd'])]
        
        if not control_topics:
            print("🎮 未发现控制topic")
            return
        
        print(f"🎮 发现 {len(control_topics)} 个控制topics")
        
        # 处理方式与传感器数据类似
        for topic in control_topics:
            topic_name = topic.replace('/', '_').strip('_')
            data_rows = []
            
            for msg_data in self.topic_data[topic]:
                msg = msg_data['data']
                timestamp = msg_data['timestamp']
                
                row = {'timestamp': timestamp}
                row.update(self._extract_numeric_fields(msg, ''))
                data_rows.append(row)
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                csv_file = self.data_dir / f"{topic_name}_control.csv"
                df.to_csv(csv_file, index=False)
                print(f"  ✅ {topic} - 导出了 {len(data_rows)} 行控制数据到 {csv_file}")
    
    def _extract_numeric_fields(self, obj, prefix: str) -> Dict[str, Any]:
        """递归提取数值字段"""
        fields = {}
        
        for attr_name in dir(obj):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr_value = getattr(obj, attr_name)
                field_name = f"{prefix}{attr_name}" if prefix else attr_name
                
                if isinstance(attr_value, (int, float)):
                    fields[field_name] = attr_value
                elif isinstance(attr_value, (list, tuple)):
                    for i, item in enumerate(attr_value):
                        if isinstance(item, (int, float)):
                            fields[f"{field_name}_{i}"] = item
                elif hasattr(attr_value, '__dict__'):
                    # 递归处理复合对象
                    sub_fields = self._extract_numeric_fields(attr_value, f"{field_name}_")
                    fields.update(sub_fields)
            except:
                continue
        
        return fields
    
    def _export_json_metadata(self):
        """导出完整的元数据到JSON"""
        metadata = {
            'bag_file': str(self.bag_path),
            'export_time': datetime.now().isoformat(),
            'topics': self.topic_info,
            'summary': {
                'total_topics': len(self.topic_info),
                'total_messages': sum(len(msgs) for msgs in self.topic_data.values()),
                'image_topics': len([t for t in self.topic_data.keys() if 'camera' in t or 'image' in t]),
                'sensor_topics': len([t for t in self.topic_data.keys() if any(k in t.lower() for k in ['imu', 'feedback', 'joint'])]),
                'control_topics': len([t for t in self.topic_data.keys() if any(k in t.lower() for k in ['control', 'target', 'command'])])
            }
        }
        
        metadata_file = self.metadata_dir / "export_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📄 元数据已导出到: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description='ROS Bag数据导出工具')
    parser.add_argument('bag_path', help='ROS bag文件路径')
    parser.add_argument('-o', '--output', default='./rosbag_export', 
                       help='输出目录 (默认: ./rosbag_export)')
    parser.add_argument('--images-only', action='store_true',
                       help='只导出图像数据')
    parser.add_argument('--data-only', action='store_true',
                       help='只导出传感器和控制数据')
    
    args = parser.parse_args()
    
    if not Path(args.bag_path).exists():
        print(f"❌ ROS bag文件不存在: {args.bag_path}")
        return
    
    exporter = RosbagDataExporter(args.bag_path, args.output)
    
    if args.images_only:
        exporter._export_topic_info()
        exporter._export_image_data()
        exporter._export_json_metadata()
    elif args.data_only:
        exporter._export_topic_info()
        exporter._export_sensor_data()
        exporter._export_control_data()
        exporter._export_json_metadata()
    else:
        exporter.export_all_data()


if __name__ == '__main__':
    main()
