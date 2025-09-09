#!/usr/bin/env python3
"""
深度分析MMK BSON文件结构
"""
import struct
from pathlib import Path
import json

def parse_bson_document(data, offset=0):
    """解析单个BSON文档"""
    if offset + 4 > len(data):
        return None, offset
    
    # 读取文档大小
    doc_size = struct.unpack('<I', data[offset:offset+4])[0]
    if offset + doc_size > len(data):
        return None, offset
    
    doc_data = data[offset:offset+doc_size]
    result = {}
    
    pos = 4  # 跳过文档大小
    
    while pos < len(doc_data) - 1:  # 最后一个字节是结束符0x00
        if pos >= len(doc_data):
            break
            
        # 读取字段类型
        field_type = doc_data[pos]
        pos += 1
        
        # 读取字段名
        name_end = doc_data.find(b'\x00', pos)
        if name_end == -1:
            break
        field_name = doc_data[pos:name_end].decode('utf-8', errors='ignore')
        pos = name_end + 1
        
        # 根据类型解析值
        if field_type == 0x01:  # double
            if pos + 8 <= len(doc_data):
                value = struct.unpack('<d', doc_data[pos:pos+8])[0]
                pos += 8
                result[field_name] = value
        elif field_type == 0x02:  # string
            if pos + 4 <= len(doc_data):
                str_len = struct.unpack('<I', doc_data[pos:pos+4])[0]
                pos += 4
                if pos + str_len <= len(doc_data):
                    value = doc_data[pos:pos+str_len-1].decode('utf-8', errors='ignore')
                    pos += str_len
                    result[field_name] = value
        elif field_type == 0x03:  # document
            if pos + 4 <= len(doc_data):
                subdoc_size = struct.unpack('<I', doc_data[pos:pos+4])[0]
                if pos + subdoc_size <= len(doc_data):
                    subdoc, _ = parse_bson_document(doc_data, pos)
                    if subdoc is not None:
                        result[field_name] = subdoc
                    pos += subdoc_size
        elif field_type == 0x04:  # array
            if pos + 4 <= len(doc_data):
                array_size = struct.unpack('<I', doc_data[pos:pos+4])[0]
                if pos + array_size <= len(doc_data):
                    array_doc, _ = parse_bson_document(doc_data, pos)
                    if array_doc is not None:
                        # 将字典转换为列表（BSON数组以索引为键）
                        array_list = []
                        for i in range(len(array_doc)):
                            if str(i) in array_doc:
                                array_list.append(array_doc[str(i)])
                        result[field_name] = array_list
                    pos += array_size
        elif field_type == 0x08:  # boolean
            if pos + 1 <= len(doc_data):
                value = doc_data[pos] != 0
                pos += 1
                result[field_name] = value
        elif field_type == 0x10:  # int32
            if pos + 4 <= len(doc_data):
                value = struct.unpack('<i', doc_data[pos:pos+4])[0]
                pos += 4
                result[field_name] = value
        elif field_type == 0x12:  # int64
            if pos + 8 <= len(doc_data):
                value = struct.unpack('<q', doc_data[pos:pos+8])[0]
                pos += 8
                result[field_name] = value
        else:
            # 未知类型，跳过
            break
    
    return result, offset + doc_size

def analyze_mmk_bson():
    """分析MMK的BSON文件"""
    data_dir = Path("/home/diy01/dev/robocoin-dataset/data/mmk/episode_0")
    
    for bson_file in data_dir.glob("*.bson"):
        print(f"=== 分析 {bson_file.name} ===")
        
        with open(bson_file, 'rb') as f:
            content = f.read()
        
        # 解析第一个文档
        doc, next_offset = parse_bson_document(content, 0)
        
        if doc:
            print("解析成功！文档结构:")
            print_structure(doc, indent=2)
            
            # 如果是数组数据，分析数组长度
            if 'frames' in doc and isinstance(doc['frames'], list):
                print(f"\n帧数据数量: {len(doc['frames'])}")
                if doc['frames']:
                    print("第一帧结构:")
                    print_structure(doc['frames'][0], indent=4)
        else:
            print("解析失败")
        
        print("\n" + "="*50 + "\n")

def print_structure(obj, indent=0):
    """打印数据结构"""
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}: {type(value).__name__}")
                if isinstance(value, list) and value:
                    print(f"{prefix}  长度: {len(value)}")
                    if isinstance(value[0], (dict, list)):
                        print(f"{prefix}  元素类型: {type(value[0]).__name__}")
                        if len(value) > 0:
                            print(f"{prefix}  第一个元素:")
                            print_structure(value[0], indent + 4)
                    else:
                        print(f"{prefix}  元素示例: {value[:3]}")
                else:
                    print_structure(value, indent + 2)
            else:
                value_str = str(value)[:50]
                if len(str(value)) > 50:
                    value_str += "..."
                print(f"{prefix}{key}: {value_str}")
    elif isinstance(obj, list):
        print(f"{prefix}列表长度: {len(obj)}")
        if obj and len(obj) > 0:
            print(f"{prefix}第一个元素:")
            print_structure(obj[0], indent + 2)

if __name__ == "__main__":
    analyze_mmk_bson()
