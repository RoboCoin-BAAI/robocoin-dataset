from pathlib import Path

import yaml
from sqlalchemy import or_

from robocoin_dataset.annotation.device_model_annotation.constant import (
    DEVICE_MODEL,
    DEVICE_MODEL_ANNOTATION_FILE_NAME,
    DEVICE_MODEL_VERSION,
)
from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.database.models import (
    DatasetDB,
    DmvAnnotationDB,
    TaskStatus,
)
from robocoin_dataset.database.services.dmv_annotation import (
    upsert_dmv_annotation_status,
)


def annotate_device_model(db_file_path: Path) -> None:
    if not db_file_path.exists():
        raise FileNotFoundError
    db = DatasetDatabase(db_file=db_file_path)
    with db.with_session() as session:
        results = (
            session.query(DatasetDB.dataset_uuid)
            .outerjoin(DmvAnnotationDB, DatasetDB.dataset_uuid == DmvAnnotationDB.dataset_uuid)
            .filter(
                or_(
                    DmvAnnotationDB.dataset_uuid.is_(None),  # 无记录
                    DmvAnnotationDB.annotation_status != TaskStatus.COMPLETED,  # 未完成
                )
            )
            .all()
        )

        for res in results:
            uuid = res[0]
            item = session.query(DatasetDB).filter(DatasetDB.dataset_uuid == uuid).first()
            print(f"yaml_file_path: {item.yaml_file_path}")
            dataset_path = Path(item.yaml_file_path).parent
            dmv_annotation_file_path = dataset_path / DEVICE_MODEL_ANNOTATION_FILE_NAME
            status = TaskStatus.FAILED
            device_model = ""
            device_model_version = ""
            if Path(dmv_annotation_file_path).exists():
                with open(dmv_annotation_file_path, "r") as f:
                    data = yaml.safe_load(f)
                    device_model = data.get(DEVICE_MODEL, "")
                    device_model_version = data.get(DEVICE_MODEL_VERSION, "")

            if device_model:
                status = TaskStatus.COMPLETED

            upsert_dmv_annotation_status(
                session=session,
                ds_uuid=item.dataset_uuid,
                annotation_status=status,
                annotatio_file_path=str(dmv_annotation_file_path),
                device_model=device_model,
                device_model_version=device_model_version,
            )
