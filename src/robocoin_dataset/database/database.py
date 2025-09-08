from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


class DatasetDatabase:
    def __init__(self, db_file: Path) -> None:
        self.db_file = db_file.expanduser().absolute()
        self.engine = None
        self.session_local = None
        self._initialize()
        self._create_tables()

    def _initialize(self) -> None:
        self.db_file = Path(self.db_file).expanduser().absolute()
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        database_url = f"sqlite:///{self.db_file}"
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def with_session(self) -> Generator[Session, None, None]:
        """
        安全的上下文管理器，确保 session 正确关闭。
        推荐在同步代码中使用。
        """
        gen = self.get_session()
        session = next(gen)
        try:
            yield session
        finally:
            gen.close()  # ✅ 触发 get_session 中的 finally

    def _create_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)
