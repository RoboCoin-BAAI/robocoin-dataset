import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler  # 可选：轮转
from pathlib import Path


def setup_logger(
    name: str,
    log_dir: str | Path,
    level=logging.INFO,  # noqa: ANN001
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    创建一个同时输出到文件（带时间戳）和控制台的日志记录器
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # 已配置，避免重复
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # 创建日志文件路径
    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Unable to mkdir log dir {log_dir}: {e}", file=sys.stderr)
        # 降级：只输出到控制台
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filepath = log_dir / f"{name}_{timestamp}.log"

    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_filepath, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"The log system has been started, log file: {log_filepath.resolve()}")

    return logger
