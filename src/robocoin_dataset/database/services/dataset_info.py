# services/yaml_import_service.py

from sqlalchemy.orm import Session

# 从你的项目中导入
from ..database import DatasetDatabase
from ..models import (
    DatasetDB,
    ObjectDB,
    SceneTypeDB,
    TaskDescriptionDB,
)

# 设置日志


def upsert_dataset_info(yaml_data: list[str, any], db: DatasetDatabase) -> None:
    """
    从 YAML 字典导入数据集到数据库。

    Args:
        yaml_data: yaml.safe_load() 解析出的字典

    Raises:
        Exception: 如果导入失败
    """
    # 初始化数据库

    # 获取会话
    session_gen = db.get_session()
    try:
        db: Session = next(session_gen)
    except StopIteration:
        raise RuntimeError("无法获取数据库会话")

    try:
        # === 1. 提取 DatasetDB 基础字段 ===
        dataset_data = {
            "dataset_name": yaml_data["dataset_name"],
            "dataset_uuid": yaml_data["dataset_uuid"],
            "device_model": yaml_data["device_model"][0],
            "end_effector_type": yaml_data["end_effector_type"],
            "operation_platform_height": yaml_data.get("operation_platform_heigth"),  # 注意拼写
        }

        # === 2. 提取多对多字段 ===
        scene_type_names = yaml_data.get("scene_type", [])
        task_description_descs = yaml_data.get("task_descriptions", [])
        yaml_objects = yaml_data.get("objects", [])

        # === 3. 处理 scene_type ===
        scene_types = []
        for name in scene_type_names:
            db_scene = db.query(SceneTypeDB).filter(SceneTypeDB.name == name).first()
            if not db_scene:
                db_scene = SceneTypeDB(name=name)
                db.add(db_scene)
            scene_types.append(db_scene)

        # === 4. 处理 task_descriptions ===
        task_descriptions = []
        for desc in task_description_descs:
            db_task = db.query(TaskDescriptionDB).filter(TaskDescriptionDB.desc == desc).first()
            if not db_task:
                db_task = TaskDescriptionDB(desc=desc)
                db.add(db_task)
            task_descriptions.append(db_task)

        # === 5. 处理 objects 及其层级（仅存字符串）===
        db_objects = []
        for obj in yaml_objects:
            if not isinstance(obj, dict):
                continue

            # 提取物体主名称（如 banana, table）
            object_name = obj["object_name"]

            # 获取或创建 ObjectDB
            db_obj = db.query(ObjectDB).filter(ObjectDB.object_name == object_name).first()
            if not db_obj:
                db_obj = ObjectDB(object_name=object_name)

            # 直接赋值字符串，不涉及层级表
            db_obj.level1_category = obj.get("level1")
            db_obj.level2_category = obj.get("level2")
            db_obj.level3_category = obj.get("level3")
            db_obj.level4_category = obj.get("level4")
            db_obj.level5_category = obj.get("level5")

            db.add(db_obj)
            db_objects.append(db_obj)
        # === 6. 查找或创建 DatasetDB ===
        db_dataset = (
            db.query(DatasetDB)
            .filter(DatasetDB.dataset_uuid == dataset_data["dataset_uuid"])
            .first()
        )

        if not db_dataset:
            db_dataset = DatasetDB(
                **{k: v for k, v in dataset_data.items() if k in DatasetDB.__table__.columns}
            )
            db.add(db_dataset)

        # 更新字段
        for k, v in dataset_data.items():
            if hasattr(db_dataset, k):
                setattr(db_dataset, k, v)

        # 关联多对多关系
        db_dataset.scene_types = scene_types
        db_dataset.task_descriptions = task_descriptions
        db_dataset.objects = db_objects

        # === 7. 提交事务 ===
        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
