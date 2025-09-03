from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


class DatasetDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().absolute()
        self.engine = None
        self.session_local = None
        self._initialize()
        self._create_tables()

    def _initialize(self) -> None:
        self.db_path = Path(self.db_path).expanduser().absolute()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        database_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    def _create_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)
