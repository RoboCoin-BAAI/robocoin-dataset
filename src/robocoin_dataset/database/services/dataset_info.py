# services/yaml_import_service.py
from sqlalchemy.orm import Session
from typing import Any, Dict

from ..models import DatasetDB, ObjectDB, SceneTypeDB, TaskDescriptionDB
import logging

logger = logging.getLogger(__name__)


def upsert_dataset_info(yaml_data: Dict[str, Any], session: Session) -> None:
    """
    将 YAML 字典写入数据库。
    调用者必须传入已打开的 SQLAlchemy Session，事务由调用者控制。
    """
    try:
        # 1. 基础字段
        dataset_data = {
            "dataset_name": yaml_data["dataset_name"],
            "dataset_uuid": yaml_data["dataset_uuid"],
            "end_effector_type": yaml_data.get("end_effector_type"),
            "operation_platform_height": yaml_data.get("operation_platform_height"),
            "yaml_file_path": yaml_data.get("yaml_file_path"),
        }
        device_model = yaml_data.get("device_model")
        if isinstance(device_model, list):
            dataset_data["device_model"] = device_model[0] if device_model else None
        else:
            dataset_data["device_model"] = device_model
        dataset_data = {k: v for k, v in dataset_data.items() if v is not None}

        # 2. 多对多字段
        scene_type_names = yaml_data.get("scene_type", [])
        task_descs = (
            yaml_data.get("task_descriptions", [])
            or yaml_data.get("task_description", [])
            or yaml_data.get("task_desc", [])
        )
        yaml_objects = yaml_data.get("objects", [])

        # 3. scene_type
        scene_types = []
        for name in scene_type_names:
            st = session.query(SceneTypeDB).filter_by(name=name).first()
            if not st:
                st = SceneTypeDB(name=name)
                session.add(st)
            scene_types.append(st)

        # 4. task_descriptions
        task_descriptions = []
        for desc in task_descs:
            td = session.query(TaskDescriptionDB).filter_by(desc=desc).first()
            if not td:
                td = TaskDescriptionDB(desc=desc)
                session.add(td)
            task_descriptions.append(td)

        # 5. objects
        db_objects = []
        for obj in yaml_objects:
            if not isinstance(obj, dict):
                continue
            name = obj.get("object_name")
            if not name:
                continue
            ob = session.query(ObjectDB).filter_by(object_name=name).first()
            if not ob:
                ob = ObjectDB(object_name=name)
            ob.level1_category = obj.get("level1")
            ob.level2_category = obj.get("level2")
            ob.level3_category = obj.get("level3")
            ob.level4_category = obj.get("level4")
            ob.level5_category = obj.get("level5")
            session.add(ob)
            db_objects.append(ob)

        # 6. dataset 本体
        ds = (
            session.query(DatasetDB)
            .filter_by(dataset_uuid=dataset_data["dataset_uuid"])
            .first()
        )
        if not ds:
            ds = DatasetDB(**dataset_data)
            session.add(ds)
        else:
            for k, v in dataset_data.items():
                setattr(ds, k, v)

        # 7. 关联
        ds.scene_types = scene_types
        ds.task_descriptions = task_descriptions
        ds.objects = db_objects

    except Exception as e:
        logger.error(f"Upsert failed for {yaml_data.get('dataset_name')}: {e}")
        raise