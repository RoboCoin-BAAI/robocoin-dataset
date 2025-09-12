#!/usr/bin/env python3
# scripts/db/import_dataset_info_from_yaml.py
import argparse
import shutil
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
import yaml
import logging
import traceback
import subprocess

# 把项目根目录塞进 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.services.dataset_info import upsert_dataset_info
from robocoin_dataset.database.check_duplicates import get_or_create_uuid, load_registry, save_registry

# 全局变量（用于本次运行去重）
used_uuids_global = set()

# 支持的 YAML 文件名
SUPPORTED_YAML_NAMES = {"local_dataset_info.yaml", "local_dataset_info.yml"}

# ------------------------------------------------------------------
# 配置日志
# ------------------------------------------------------------------
def setup_logging(log_dir: Path, dry_run: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "import_dataset_info.log"
    if log_file.exists() and not dry_run:
        log_file.unlink()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)

    if not dry_run:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    logging.info(f"日志已启动，日志文件：{log_file}")

# ------------------------------------------------------------------
# 使用 os.walk 查找 YAML 文件（跳过子目录）
# ------------------------------------------------------------------
def find_local_yaml_files(root: Path) -> List[Path]:
    found = []
    for current, dirnames, files in os.walk(root):
        if SUPPORTED_YAML_NAMES & set(files):
            filename = next(iter(SUPPORTED_YAML_NAMES & set(files)))
            found.append(Path(current) / filename)
            dirnames.clear()
    return found

# ------------------------------------------------------------------
# 数据清理函数
# ------------------------------------------------------------------
def clean_data_value(value: Any) -> Any:
    if isinstance(value, list):
        return str(value[0]) if len(value) == 1 else str(value) if value else None
    elif isinstance(value, dict):
        return str(value)
    return value

# ------------------------------------------------------------------
# 读取 + 检查 dataset_uuid（不再回写 local_dataset_info.yaml）
# ------------------------------------------------------------------
def load_and_patch(yaml_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    try:
        with yaml_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"解析失败 {yaml_path}: {e}")
        return {}

    if "dataset_name" not in data:
        logging.warning(f"跳过无效文件（缺少 dataset_name）：{yaml_path}")
        return {}

    existing_uuid = data.get("dataset_uuid")
    if existing_uuid:
        used_uuids_global.add(existing_uuid)

    # 如果 local 文件里没有，尝试读取同目录 dataset_uuid.yaml
    if not existing_uuid:
        sidecar = yaml_path.with_name("dataset_uuid.yaml")
        if sidecar.exists():
            try:
                side_data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
                existing_uuid = side_data.get("dataset_uuid")
                if existing_uuid:
                    data["dataset_uuid"] = existing_uuid
                    used_uuids_global.add(existing_uuid)
                    logging.info(f"从 {sidecar} 读取到 UUID: {existing_uuid}")
            except Exception as e:
                logging.warning(f"读取 {sidecar} 失败: {e}")

    # 仍然拿不到则生成新 UUID，仅写入 dataset_uuid.yaml
    if not existing_uuid:
        task_desc = data.get("task_description") or data.get("task_desc") or "unknown_task"
        device_model = data.get("device_model") or "unknown_device"

        try:
            registry_file = PROJECT_ROOT / "dataset_registry.yaml"
            new_uuid = get_or_create_uuid(
                task=task_desc,
                device=device_model,
                yaml_path=str(yaml_path),
                registry_file=str(registry_file),
                used_uuids=used_uuids_global
            )
            data["dataset_uuid"] = new_uuid
            used_uuids_global.add(new_uuid)

            sidecar = yaml_path.with_name("dataset_uuid.yaml")
            if not dry_run:
                sidecar.write_text(
                    yaml.dump({"dataset_uuid": new_uuid}, indent=2, allow_unicode=True),
                    encoding="utf-8",
                )
                logging.info(f"已写入新 UUID 到 {sidecar}: {new_uuid}")
            else:
                logging.info(f"[dry-run] 将写入新 UUID 到 {sidecar}: {new_uuid}")

        except Exception as e:
            logging.error(f"自动生成 UUID 失败 {yaml_path}: {e}")
            return {}

    # 清理字段
    for key in ('device_model', 'end_effector_type', 'operation_platform_height'):
        if key in data:
            data[key] = clean_data_value(data[key])

    logging.info(f"[{data['dataset_name']}] -> {data['dataset_uuid']}")
    data['yaml_file_path'] = str(yaml_path)
    return data

# ---------------- 收集模式 ----------------
def collect_yaml_files(root_dirs: List[Path], output_dir: Path, dry_run: bool = False) -> None:
    log_dir = output_dir / "logs"
    setup_logging(log_dir, dry_run=dry_run)

    if not dry_run:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        logging.info(f"[dry-run] 输出目录: {output_dir} (将被清空或创建)")

    hub = {}
    idx = 0
    for r in root_dirs:
        for dirpath, dirnames, files in os.walk(r):
            if SUPPORTED_YAML_NAMES & set(files):
                filename = next(iter(SUPPORTED_YAML_NAMES & set(files)))
                src = Path(dirpath) / filename
                dst = output_dir / f"local_dataset_info_{idx}.yml"
                if not dry_run:
                    shutil.copy2(src, dst)
                hub[str(dst.resolve())] = str(src.resolve())
                logging.info(f"已复制: {src} → {dst}")
                idx += 1
                dirnames.clear()

    if not dry_run:
        hub_file = output_dir / "local_dataset_info_hub.yml"
        hub_file.write_text(
            yaml.dump(hub, sort_keys=True, allow_unicode=True, indent=2),
            encoding="utf-8",
        )
        logging.info(f"已生成 hub 文件: {hub_file}")

    logging.info(f"已收集 {idx} 个文件到 {output_dir}")

# ------------------------------------------------------------------
# 批量处理函数
# ------------------------------------------------------------------
def process_files_batch(yaml_files: List[Path], max_workers: int = 8, dry_run: bool = False) -> tuple[List[Dict[str, Any]], int]:
    datasets = []
    invalid_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(load_and_patch, yml, dry_run): yml for yml in yaml_files}
        for future in concurrent.futures.as_completed(future_to_file):
            yml = future_to_file[future]
            try:
                data = future.result()
                if data:
                    datasets.append(data)
                else:
                    invalid_count += 1
            except Exception as e:
                logging.error(f"处理文件失败 {yml}: {e}")
                invalid_count += 1

    return datasets, invalid_count

# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main() -> None:
    global used_uuids_global
    used_uuids_global = set()

    parser = argparse.ArgumentParser(
        description="递归查找 local_dataset_info.yaml/.yml，检查 dataset_uuid 并批量入库"
    )
    parser.add_argument(
        "scan_root",
        type=str,
        help="要扫描的根目录，支持多个路径（用空格、逗号、分号分隔）",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./db/datasets.db",
        help="数据库文件路径，默认 ./db/datasets.db",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行工作线程数，默认8",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="仅收集 yaml 文件，不做数据库导入"
    )
    parser.add_argument(
        "--collect-output",
        type=Path,
        default=Path("./collected_yamls"),
        help="收集模式下的输出目录（日志也存放于此）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描和模拟，不写入文件或数据库"
    )
    args = parser.parse_args()

    # 解析多个路径
    root_paths = re.split(r"[\s;,:\n]+", args.scan_root.strip())
    root_paths = [Path(p.strip()) for p in root_paths if p.strip()]

    # 设置日志
    log_dir = args.collect_output / "logs"
    setup_logging(log_dir, dry_run=args.dry_run)

    if args.collect_only:
        logging.info(f"收集模式：从 {len(root_paths)} 个根目录收集 YAML 文件")
        collect_yaml_files(root_paths, args.collect_output, dry_run=args.dry_run)
        return

    # 扫描所有路径
    all_yaml_files = []
    for root in root_paths:
        if not root.exists():
            logging.error(f"路径不存在：{root}")
            continue
        logging.info(f"扫描目录: {root}")
        yaml_files = find_local_yaml_files(root)
        all_yaml_files.extend(yaml_files)

    if not all_yaml_files:
        logging.warning("未找到任何 local_dataset_info.yaml/.yml，退出。")
        sys.exit(0)

    logging.info(f"找到 {len(all_yaml_files)} 个 YAML 文件")

    # 并行处理
    datasets, invalid_count = process_files_batch(all_yaml_files, args.workers, dry_run=args.dry_run)

    if invalid_count > 0:
        logging.warning(f"跳过 {invalid_count} 个无效文件")

    if not datasets:
        logging.warning("没有有效数据集，退出。")
        sys.exit(0)

    logging.info(f"准备处理 {len(datasets)} 个数据集")

    if args.dry_run:
        logging.info("[dry-run] 模拟结束，未写入数据库或文件。")
        return

    # 1. 创建数据库路径
    db_path = Path(args.db_path).expanduser().absolute()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. 创建数据库实例
    db = DatasetDatabase(db_path)
    logging.info(f"数据库连接已创建: {db_path}")

    # 3. 写入数据库
    try:
        with db.with_session() as session:
            for record in datasets:
                upsert_dataset_info(record, session=session)
            session.commit()
        success_count = len(datasets)
        logging.info(f"成功导入 {success_count} 个数据集到数据库")

       # logging.info("正在执行 separate.py 脚本...")
       # subprocess.run([sys.executable, "separate.py"], check=True)
        logging.info("separate.py 脚本执行完毕。")

    except subprocess.CalledProcessError as e:
        logging.critical(f"separate.py 脚本执行失败: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"批量导入失败: {e}")
        logging.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
    

"""
测试
python scripts/db/import_dataset_info_from_yaml.py \
       /data/test \
       --dry-run
       
收集文件
python scripts/db/import_dataset_info_from_yaml.py \
       data --collect-only \
       --collect-output ./yaml_backup

扫描入库
python scripts/db/import_dataset_info_from_yaml.py \
       /mnt/nas/4unitree_g1/basket_storage_apple \
       --workers 16 \
       --db-path /home/adminpc1/Desktop/3/datasets.db
"""