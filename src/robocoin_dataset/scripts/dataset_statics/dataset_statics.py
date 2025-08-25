from dataclasses import dataclass, field

import yaml


@dataclass
class TaskObject:
    object_name: str = ""
    level_category: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"object_name": self.object_name, "level_category": self.level_category}


@dataclass
class DatasetStatics:
    dataset_name: str = ""
    dataset_path: str = ""
    device_model: str = ""
    end_effector_type: str = ""
    atomic_actions: list[str] = field(default_factory=list)
    objects: list[TaskObject] = field(default_factory=list)
    episode_frames_num: list[int] = field(default_factory=list)

    def _to_yaml(self) -> str:
        # 手动将对象转为字典，确保嵌套的 TaskObject 调用 to_dict()
        data = {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "device_model": self.device_model,
            "end_effector_type": self.end_effector_type,
            "atomic_actions": self.atomic_actions,
            "episode_frames_num": self.episode_frames_num,
            "objects": [
                obj.to_dict() for obj in self.objects
            ],  # ✅ 调用每个 TaskObject 的 to_dict()
        }
        print(data["objects"])
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
