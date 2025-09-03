from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Column,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# =====================
# 枚举类型
# =====================


class ConvertStatus(str, PyEnum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# =====================
# 主表：DatasetDB
# =====================


class DatasetDB(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(255), unique=False, index=True, nullable=False)

    # ✅ 核心：dataset_uuid 作为全局唯一业务标识
    dataset_uuid = Column(String(255), unique=True, index=True, nullable=False)

    device_model = Column(String(100), nullable=False)
    end_effector_type = Column(String(100), nullable=False)
    operation_platform_height = Column(Float, nullable=True)

    # 多对多关系
    scene_types = relationship(
        "SceneTypeDB", secondary="dataset_scene_types", back_populates="datasets"
    )
    task_descriptions = relationship(
        "TaskDescriptionDB", secondary="dataset_task_descriptions", back_populates="datasets"
    )
    objects = relationship("ObjectDB", secondary="dataset_objects", back_populates="datasets")


# =====================
# 分类与多对多表
# =====================


class SceneTypeDB(Base):
    __tablename__ = "scene_types"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)

    # 反向关系
    datasets = relationship(
        "DatasetDB", secondary="dataset_scene_types", back_populates="scene_types"
    )


class TaskDescriptionDB(Base):
    __tablename__ = "task_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    desc = Column(String(255), unique=True, index=True, nullable=False)

    # 反向关系：注意是 "datasets"，不是 "task_descriptions"
    datasets = relationship(
        "DatasetDB", secondary="dataset_task_descriptions", back_populates="task_descriptions"
    )


class ObjectDB(Base):
    __tablename__ = "object"
    id = Column(Integer, primary_key=True, index=True)
    object_name = Column(String(100), nullable=False, index=True)

    level1_category = Column(String(100), unique=False, nullable=True)
    level2_category = Column(String(100), unique=False, nullable=True)
    level3_category = Column(String(100), unique=False, nullable=True)
    level4_category = Column(String(100), unique=False, nullable=True)
    level5_category = Column(String(100), unique=False, nullable=True)

    # 反向关系：ObjectDB -> DatasetDB 多对多
    datasets = relationship("DatasetDB", secondary="dataset_objects", back_populates="objects")


# =====================
# 多对多关联表
# =====================

dataset_scene_types = Table(
    "dataset_scene_types",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("scene_type_id", Integer, ForeignKey("scene_types.id"), primary_key=True),
)

dataset_task_descriptions = Table(
    "dataset_task_descriptions",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("task_description_id", Integer, ForeignKey("task_descriptions.id"), primary_key=True),
)

dataset_objects = Table(
    "dataset_objects",
    Base.metadata,
    Column("dataset_id", Integer, ForeignKey("datasets.id"), primary_key=True),
    Column("object_id", Integer, ForeignKey("object.id"), primary_key=True),
)


# =====================
# 独立状态表：使用 dataset_uuid 关联
# =====================


class AnnotationDB(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)

    # ✅ 使用 dataset_uuid 作为关联字段
    dataset_uuid = Column(String(255), index=True, nullable=False)

    # 标注状态
    annotation_status = Column(Boolean, default=False, nullable=False)

    # ✅ 唯一约束：确保一个数据集最多一条标注记录
    __table_args__ = (UniqueConstraint("dataset_uuid", name="uix_dataset_uuid_annotation"),)


class LeFormatConvertDB(Base):
    __tablename__ = "lerobot_format_convert"

    id = Column(Integer, primary_key=True, index=True)

    # ✅ 使用 dataset_uuid 作为关联字段
    dataset_uuid = Column(String(255), index=True, nullable=False)

    convert_status = Column(Enum(ConvertStatus), default=ConvertStatus.pending, nullable=False)
    convert_path = Column(String(255), nullable=True)  # 移除 unique=True，允许多个不同路径

    # ✅ 唯一约束：一个 uuid 最多一个转换记录
    __table_args__ = (UniqueConstraint("dataset_uuid", name="uix_dataset_uuid_convert"),)


class EpisodeFrameDB(Base):
    __tablename__ = "episode_frames"

    id = Column(Integer, primary_key=True, index=True)

    # 存储 dataset_uuid（字符串格式）
    dataset_uuid = Column(String(255), index=True, nullable=False)  # 改为 255，与 datasets 表一致

    episode_index = Column(Integer, index=True, nullable=False)
    frames_num = Column(Integer, nullable=False)
