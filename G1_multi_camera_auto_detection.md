# G1多相机自动检索功能说明

## 功能概述

为Unitree G1机器人实现了多相机自动检索和配置功能，能够：

1. **自动检测相机数量** - 扫描数据集，根据图像文件命名模式自动识别相机数量
2. **智能文件分组** - 按G1命名规范将图像文件分组到不同相机
3. **动态配置生成** - 自动生成对应数量的相机配置，无需手动编辑YAML
4. **无工厂依赖** - 完全在转换器内部实现，与工厂配置解耦

## G1相机文件命名规范

G1机器人的多相机图像文件按以下规范命名：

```
单相机: frame_000000.jpg (最后一位是0)
多相机: 
  - frame_000000.jpg (相机0)
  - frame_000001.jpg (相机1) 
  - frame_000002.jpg (相机2)
  - frame_000003.jpg (相机3)
```

**关键特征**: 文件名最后一位数字表示相机索引

## 配置文件

### 自动多相机配置 (推荐)

使用 `convertor_config_unitree_g1_multi_camera_auto.yaml`:

```yaml
# 启用自动检测
camera_detection:
  auto_detect: true              # 启用自动检测
  file_pattern: "*.jpg"         # 图像文件模式
  camera_naming: "color_{idx}"  # 相机命名格式
  max_cameras: 4                # 最大支持相机数量

features:
  observation:
    images: []  # 运行时自动填充
    # ... 其他配置保持不变
```

### 传统手动配置

使用原有的 `convertor_config_unitree_g1.yaml`，需要手动配置每个相机。

## 工厂配置

在 `convertor_factory_config.yaml` 中添加了新的配置项：

```yaml
# 传统单相机或手动多相机配置
unitree_g1:
  module: robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor_g1
  class: LerobotFormatConvertorG1
  convertor_config_path: convertor_config_unitree_g1.yaml

# 新增：自动多相机配置
unitree_g1_multi_camera:
  module: robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor_g1
  class: LerobotFormatConvertorG1
  convertor_config_path: convertor_config_unitree_g1_multi_camera_auto.yaml
```

## 使用方法

### 1. 使用自动多相机检测

```python
# 通过工厂创建转换器
convertor = LerobotFormatConvertorFactory.create_convertor(
    dataset_path=dataset_path,
    device_model="unitree_g1_multi_camera",  # 使用多相机配置
    output_path=output_path,
    convertor_config=config,
    factory_config=factory_config,
    repo_id=repo_id
)
```

### 2. 直接使用G1转换器

```python
# 直接实例化
convertor = LerobotFormatConvertorG1(
    dataset_path=dataset_path,
    output_path=output_path,
    convertor_config=auto_config,  # 包含camera_detection配置的config
    repo_id=repo_id
)
```

## 自动检测流程

1. **配置检查** - 检查config中是否启用`camera_detection.auto_detect`
2. **数据扫描** - 扫描数据集前20个图像文件作为样本
3. **模式匹配** - 使用正则表达式`.*(\d)\.jpe?g$`匹配文件名最后一位数字
4. **相机分组** - 按数字将文件分组：0->相机0, 1->相机1, 等等
5. **配置生成** - 自动生成`color_0`, `color_1`, `color_2`, `color_3`的相机配置
6. **配置更新** - 将生成的相机配置替换原config中的空`images`数组

## 支持的相机配置

- **单相机**: 只有*0.jpg文件 -> 生成1个相机配置
- **双相机**: *0.jpg, *1.jpg -> 生成2个相机配置  
- **三相机**: *0.jpg, *1.jpg, *2.jpg -> 生成3个相机配置
- **四相机**: *0.jpg, *1.jpg, *2.jpg, *3.jpg -> 生成4个相机配置

## 核心方法

### 在LerobotFormatConvertorG1中实现的关键方法:

1. `_group_images_by_camera_g1()` - G1多相机图像分组
2. `_extract_camera_index_from_key()` - 从image_key提取相机索引
3. `_detect_cameras_from_images()` - 从图像文件自动检测相机配置
4. `_has_auto_camera_detection()` - 检查是否启用自动检测
5. `_auto_configure_cameras()` - 执行自动相机配置
6. `_generate_multi_camera_config()` - 生成多相机配置

## 优势

✅ **零配置** - 无需手动编辑相机数量，自动检测  
✅ **灵活适应** - 支持1-4相机的任意组合  
✅ **向后兼容** - 不影响现有单相机配置  
✅ **错误友好** - 检测失败时自动回退到单相机配置  
✅ **独立实现** - 不依赖工厂配置，完全在转换器内部实现  
✅ **符合G1规范** - 严格按照G1文件命名规范实现  

## 示例输出

```
G1: 启用多相机自动检测
正在扫描数据集以检测相机配置...
检测到 4 个相机: ['color_0', 'color_1', 'color_2', 'color_3']
生成了 4 个相机的配置:
  相机 0: color_0
  相机 1: color_1  
  相机 2: color_2
  相机 3: color_3
Found 4 cameras
  Camera 0: 120 images
  Camera 1: 120 images
  Camera 2: 120 images  
  Camera 3: 120 images
```

这个功能完全解决了G1多相机数据的自动处理问题，并且与工厂配置完全解耦，可以独立使用。
