# Episode子目录结构问题解决方案

## 问题描述

在处理数据集时遇到的问题：
- 有的一个任务有很多episode，放在task_info.yaml文件夹的子目录下
- 但是互相也不在同一个文件夹中，如episode1文件夹和episode2文件夹
- 现有的递归搜索逻辑(`rglob("*.json")`)会找到所有JSON文件，但可能顺序不对

## 解决方案

在JPG+JSON转换器(`lerobot_format_convertor_g1.py`)中实现了智能episode目录发现功能，**不修改基类代码**。

### 核心功能

1. **智能Episode目录识别**
   ```python
   def _find_episode_directories_g1(self, task_path: Path) -> list[Path]:
   ```
   - 支持多种命名格式：`episode1`, `episode_1`, `ep1`, `ep_1`, `001`, `1`等
   - 使用正则表达式模式匹配
   - 递归搜索子目录（限制深度避免过深搜索）
   - 验证目录中包含JSON文件

2. **智能排序**
   - 按episode编号数值排序
   - 相同编号时按目录名排序
   - 确保episode1, episode2, episode_3, ep4, 005的正确顺序

3. **主JSON文件选择**
   ```python
   def _select_main_json_file(self, json_files: list[Path]) -> Path:
   ```
   - 优先选择包含"data"的文件
   - 其次选择包含"episode"的文件
   - 最后按文件名长度和字母顺序选择

4. **JPG文件递归搜索**
   ```python
   def _prepare_episode_images_buffer(self, task_path: Path, ep_idx: int) -> any:
   ```
   - 从JSON文件同级目录递归搜索JPG文件
   - 支持多种格式：`.jpg`, `.JPG`, `.jpeg`, `.JPEG`
   - 支持多级目录结构
   - 按文件名自然排序

### 使用方式

在`task_episode_jsonfile_paths`方法中：

```python
@cached_property
def task_episode_jsonfile_paths(self) -> dict[Path, list[Path]]:
    task_episode_paths = {}
    for path in self.path_task_dict.keys():
        if path.exists():
            # 方法1: 尝试智能episode目录发现
            episode_dirs = self._find_episode_directories_g1(path)
            
            if episode_dirs:
                # 找到有规律的episode目录
                json_files = []
                for episode_dir in episode_dirs:
                    episode_json_files = list(episode_dir.rglob("*.json"))
                    if episode_json_files:
                        main_file = self._select_main_json_file(episode_json_files)
                        json_files.append(main_file)
                
                task_episode_paths[path] = json_files
                continue
            
            # 方法2: 回退到递归搜索
            json_files = natsorted(list(path.rglob("*.json")))
            task_episode_paths[path] = json_files
    
    return task_episode_paths
```

### 支持的目录结构示例

```
task_root/
├── local_task_info.yaml
├── episode1/                 # 直接在根目录
│   ├── episode_data.json
│   ├── images/              # 图像可以在多级目录中
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── subdir/
│   │       └── frame_000002.jpg
│   ├── data/
│   │   └── captures/
│   │       └── frame_000003.jpg
│   └── frame_000004.jpg     # 图像也可以直接在episode目录下
├── episode2/                 # 直接在根目录
│   ├── episode_data.json
│   └── images/
├── sub_episodes/             # 在子目录中
│   └── episode_3/
│       ├── episode_data.json
│       └── images/
├── data/                     # 在其他子目录中
│   └── ep4/
│       ├── episode_data.json
│       └── deep/
│           └── nested/
│               └── path/
│                   └── frame_000005.jpeg
└── collected_data/           # 数字格式
    └── 005/
        ├── episode_data.json
        └── images/
            └── other_image.JPG
```

### JPG文件搜索规则

1. **搜索范围**: 从JSON文件同级目录开始递归搜索
2. **支持格式**: `.jpg`, `.JPG`, `.jpeg`, `.JPEG`
3. **搜索深度**: 递归搜索所有子目录
4. **排序方式**: 按文件名自然排序（frame_000000.jpg < frame_000001.jpg < frame_000010.jpg）
5. **图像访问**: 按索引访问，`frame_idx=0`对应第一张图片，`frame_idx=1`对应第二张图片

### 发现结果

智能发现会按正确顺序找到：
1. episode1 (编号1)
2. episode2 (编号2)  
3. episode_3 (编号3)
4. ep4 (编号4)
5. 005 (编号5)

## 优势

✅ **自动识别多种episode命名格式**  
✅ **按数字顺序正确排序episode**  
✅ **支持episode分散在不同子目录**  
✅ **智能选择主要JSON数据文件**  
✅ **兼容现有递归搜索方案**  
✅ **不需要修改基类代码**  
✅ **深度搜索但限制层数避免性能问题**  
✅ **验证目录有效性（包含JSON文件）**  
✅ **从JSON同级目录递归搜索JPG文件**  
✅ **支持多种图像格式和多级目录结构**  
✅ **图像文件按名称自然排序**  
✅ **灵活的图像文件组织方式**  

## 后备兼容性

如果智能发现没有找到episode目录，会自动回退到原有的递归搜索：
```python
json_files = natsorted(list(path.rglob("*.json")))
```

这确保了与现有数据集的完全兼容性。

## 测试验证

创建了完整的测试套件验证功能：
- `test_g1_episode_discovery.py`: 基本episode发现功能测试
- `realistic_episode_structure_demo.py`: 真实场景演示
- `test_jpg_recursive_search.py`: JPG文件递归搜索功能测试

测试覆盖：
- 多种episode命名格式
- 分散子目录结构
- 正确排序验证
- JSON文件选择逻辑
- JPG文件递归搜索
- 多种图像格式支持
- 多级目录结构
- 图像读取和处理
- 后备方案兼容性
