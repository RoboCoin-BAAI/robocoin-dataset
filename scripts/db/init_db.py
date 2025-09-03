import argparse

# 添加项目根目录到 sys.path，确保能导入 models
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from robocoin_dataset.database.database import DatasetDatabase


def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="初始化数据库")
    parser.add_argument("--db-dir", type=str, default="db", help="数据库文件所在目录（默认: ./db）")
    parser.add_argument(
        "--db-name", type=str, default="datasets.db", help="数据库文件名（默认: datasets.db）"
    )

    args = parser.parse_args()

    # 构建数据库文件路径
    db_path = Path(args.db_dir) / args.db_name
    print(f"database path: {db_path}")

    # 初始化数据库引擎
    db = DatasetDatabase(db_path=db_path)

    # 创建所有表

    print("✅ 数据库表已成功创建！")
    print(f"💾 数据库文件位置: {db_path}")


if __name__ == "__main__":
    main()
