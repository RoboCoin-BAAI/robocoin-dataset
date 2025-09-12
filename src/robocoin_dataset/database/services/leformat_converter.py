from datetime import datetime

from sqlalchemy.orm import Session

from robocoin_dataset.database.models import LeFormatConvertDB, TaskStatus


def upsert_leformat_convert(
    session: Session,
    ds_uuid: str,
    convert_status: TaskStatus,
    update_message: str | None = None,
    leformat_path: str | None = None,
) -> None:
    """
    Upsert LeFormatConvertDB 记录。
    :param session: 已打开的 SQLAlchemy Session（由调用方管理生命周期）
    :param ds_uuid: 数据集 UUID
    :param convert_status: 转换状态
    :param update_message: 可选，更新消息
    :param leformat_path: 可选，输出路径
    """
    try:
        # 查询是否存在
        item = (
            session.query(LeFormatConvertDB)
            .filter(LeFormatConvertDB.dataset_uuid == ds_uuid)
            .first()
        )

        if item is None:
            # 创建新记录
            item = LeFormatConvertDB(
                dataset_uuid=ds_uuid,
                convert_status=convert_status,
                convert_path=leformat_path,
                update_message=update_message,
                updated_at=datetime.now(),
            )
        else:
            # 更新现有记录
            item.convert_status = convert_status
            item.updated_at = datetime.now()
            if update_message is not None:
                item.err_message = update_message
            if leformat_path is not None:
                item.convert_path = leformat_path

        session.add(item)
        session.commit()

    except Exception as e:
        session.rollback()
        raise RuntimeError(
            f"Failed to upsert LeFormatConvertDB record for dataset {ds_uuid}: {e}"
        ) from e
