from dataclasses import dataclass, field

import yaml


@dataclass
class TaskObject:
    object_name: str = ""
    level_category: dict[str:str] = field(default_factory=dict)


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
        return yaml.dump(self.__dict__)
