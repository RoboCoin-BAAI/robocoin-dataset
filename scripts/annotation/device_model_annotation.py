import argparse
from pathlib import Path

from robocoin_dataset.annotation.device_model_annotation.device_model_annotation import (
    annotate_device_model,
)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "db_file_path",
        type=Path,
        help="Path to the dataset (default: )",
    )

    args = argparser.parse_args()
    annotate_device_model(db_file_path=args.db_file_path)

""" usage:
python scripts/annotation/device_model_annotation.py ./db/datasets.db
"""
