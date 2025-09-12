from sqlalchemy.orm import Session

from robocoin_dataset.database.models import (
    SubtaskAnnotationDB,
    TaskStatus,
)

from ..models import (
    TaskStatus,
)

# class EpisodeSubtaskAnnotation:
#     ep_idx: int
#     start_frame_idx: list[int]
#     end_frame_idx: list[int]
#     annotations: list[str]

#     def __init__(
#         self,
#         ep_idx: int,
#         start_frame_idx: list[int],
#         end_frame_idx: list[int],
#         annotations: list[str],
#     ) -> None:
#         self.ep_idx = ep_idx
#         self.start_frame_idx = start_frame_idx
#         self.end_frame_idx = end_frame_idx
#         self.annotations = annotations

#     def validate(self) -> None:
#         if len(self.start_frame_idx) != len(self.end_frame_idx):
#             raise ValueError("start_frame_idx and end_frame_idx must have the same length")
#         if len(self.start_frame_idx) != len(self.annotations):
#             raise ValueError("start_frame_idx and annotations must have the same length")


def upsert_subtask_annotation_status(
    session: Session,
    ds_uuid: str,
    subtask_annotation_file_path: str = "",
    subtask_annotation_status: TaskStatus = TaskStatus.PROCESSING,
    err_msg: str = "",
) -> None:
    """
    Upsert a subtask annotation by episode_index using DatasetDatabase.with_session.

    Args:
        ds_uuid: Dataset UUID
        ep_idx: Episode index (unique key)
        start_frame_idx: Start frame index
        end_frame_idx: End frame index
        subtask_annotation: Annotation label
        db_instance: An instance of DatasetDatabase
    """

    try:
        # 查询是否存在
        item = (
            session.query(SubtaskAnnotationDB)
            .filter(SubtaskAnnotationDB.dataset_uuid == ds_uuid)
            .first()
        )

        if item is None:
            # 创建新记录
            item = SubtaskAnnotationDB(
                dataset_uuid=ds_uuid,
            )
        item.annotation_status = subtask_annotation_status
        item.annotatio_file_path = subtask_annotation_file_path
        item.err_message = err_msg

        session.add(item)
        session.commit()

    except Exception as e:
        session.rollback()
        raise RuntimeError(
            f"Failed to upsert subtask annotation status for dataset {ds_uuid}: {e}"
        ) from e
