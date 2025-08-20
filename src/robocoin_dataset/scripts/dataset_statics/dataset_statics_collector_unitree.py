import json
import os
from pathlib import Path

from .dataset_statics_collector import DatasetStaticsCollector


class DatasetStaticsCollectorUnitree(DatasetStaticsCollector):
    def __init__(self, dataset_info_file: Path, dataset_dir: Path) -> None:
        super().__init__(dataset_info_file, dataset_dir)

    def _collect_episode_frames_num(self) -> None:
        """
        Collect episode frames number from the dataset path.
        """
        for root, _, files in os.walk(self.dataset_dir):
            if "data.json" not in files:
                continue
            del _[:]  

            data_file = Path(root) / "data.json"
            max_idx = -1

            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("{"):
                        try:
                            obj = json.loads(line)
                            idx = obj.get("idx")
                            if isinstance(idx, int):
                                max_idx = max(max_idx, idx)
                        except json.JSONDecodeError:
                            continue
                    else:
                        if "idx" in line:
                            parts = line.split()
                            if len(parts) > 2:
                                try:
                                    idx = int(parts[2])
                                    max_idx = max(max_idx, idx)
                                except ValueError:
                                    continue
            self.episode_frame_cnt[str(data_file.parent)] = max_idx + 1