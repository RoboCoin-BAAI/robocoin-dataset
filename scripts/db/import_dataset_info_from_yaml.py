#!/usr/bin/env python3
# scripts/import_yaml.py

import argparse
import sys
from pathlib import Path

import yaml

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.services.dataset_info import upsert_dataset_info

# 将项目根目录加入 sys.path，以便导入本地模块
# 假设你的项目结构如下：
# project_root/
# ├── scripts/
# │   └── import_yaml.py
# ├── services/
# │   └── yaml_import_service.py
# ├── database.py
# └── models.py
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入你之前定义的函数


def main() -> None:
    parser = argparse.ArgumentParser(description="将 YAML 配置文件导入机器人数据集数据库")
    parser.add_argument("yaml_file", type=Path, help="YAML 配置文件路径（例如：config.yaml）")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./db/datasets.db",
        help="数据库文件文件路径，默认为(./db/datasets.db)",
    )

    args = parser.parse_args()

    # 构建数据库文件路径
    db_path = Path(args.db_path)

    # 检查文件是否存在
    if not args.yaml_file.exists():
        print(f"❌ 错误：文件 '{args.yaml_file}' 不存在。", file=sys.stderr)
        sys.exit(1)

    if args.yaml_file.suffix.lower() not in [".yaml", ".yml"]:
        print(f"⚠️  警告：文件 '{args.yaml_file}' 后缀不是 .yaml 或 .yml，但仍尝试解析。")

    # 读取 YAML 文件
    try:
        with open(args.yaml_file, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 读取或解析 YAML 文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 验证必要字段
    required_fields = ["dataset_name", "dataset_uuid"]
    for field in required_fields:
        if field not in yaml_data:
            print(f"❌ YAML 文件缺少必要字段: '{field}'", file=sys.stderr)
            sys.exit(1)

    # 调用导入函数
    try:
        print(f"📌 正在导入 YAML 文件: {args.yaml_file}")
        # 初始化数据库引擎
        db = DatasetDatabase(db_path=db_path)
        print(f"💾 数据库文件位置: {db_path}")

        upsert_dataset_info(yaml_data, db)
        print(f"🎉 成功导入数据集 '{yaml_data['dataset_name']}' 到数据库。")
    except Exception as e:
        print(f"❌ 导入失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
