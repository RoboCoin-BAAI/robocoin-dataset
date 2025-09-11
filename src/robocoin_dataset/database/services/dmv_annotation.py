from sqlalchemy.orm import Session

from robocoin_dataset.database.models import (
    DmvAnnotationDB,
    TaskStatus,
)


def upsert_dmv_annotation_status(
    session: Session,
    ds_uuid: str,
    annotation_status: TaskStatus,
    annotatio_file_path: str,
    device_model: str,
    device_model_version: str,
) -> None:
    try:
        item = (
            session.query(DmvAnnotationDB).filter(DmvAnnotationDB.dataset_uuid == ds_uuid).first()
        )
        if item is None:
            item = DmvAnnotationDB(
                dataset_uuid=ds_uuid,
                annotation_status=annotation_status,
            )

        item.annotatio_file_path = annotatio_file_path
        item.annotation_status = annotation_status
        item.device_model = device_model
        item.device_model_version = device_model_version
        session.add(item)
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError(
            f"Failed to upsert device model version annotation for dataset {ds_uuid}: {e}"
        ) from e
