from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


DEFAULT_TAG_TYPES = [
    ("hf_url", "HuggingFace", "Ссылка на модель на HuggingFace"),
    ("license", "Лицензия", "Лицензия модели"),
    ("api", "API", "API, через которое тестировалась модель"),
    ("engine", "Движок", "Версия движка инференса"),
    ("params", "Параметры", "Количество параметров модели"),
    ("notes", "Заметки", "Дополнительные заметки"),
]


class Base(DeclarativeBase):
    pass


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class TagTypeRecord(Base):
    __tablename__ = "tag_types"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))


class ModelTagRecord(Base):
    __tablename__ = "model_tags"

    model_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("models.id", ondelete="CASCADE"),
        primary_key=True,
    )
    key: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("tag_types.key", ondelete="CASCADE"),
        primary_key=True,
    )
    value: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))


class Database:
    """Обёртка над PostgreSQL для хранения метаданных лидерборда."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def connect(self):
        self.engine = create_async_engine(
            self.database_url,
            pool_pre_ping=True,
            future=True,
        )
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def close(self):
        if self.engine:
            await self.engine.dispose()

    async def ensure_default_tag_types(self):
        """Гарантирует наличие стандартных типов тегов после миграций."""
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            for key, label, description in DEFAULT_TAG_TYPES:
                stmt = insert(TagTypeRecord).values(
                    key=key,
                    label=label,
                    description=description,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[TagTypeRecord.key],
                    set_={
                        "label": stmt.excluded.label,
                        "description": stmt.excluded.description,
                    },
                )
                await session.execute(stmt)

    async def ensure_model(self, model_id: str, display_name: str):
        """Создаёт модель или обновляет её отображаемое имя."""
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            stmt = insert(ModelRecord).values(id=model_id, display_name=display_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ModelRecord.id],
                set_={"display_name": stmt.excluded.display_name},
            )
            await session.execute(stmt)

    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            row = await session.get(ModelRecord, model_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "display_name": row.display_name,
                "enabled": row.enabled,
                "created_at": row.created_at.isoformat(),
            }

    async def get_all_models(self) -> List[Dict[str, Any]]:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            rows = await session.scalars(select(ModelRecord).order_by(ModelRecord.display_name))
            return [
                {
                    "id": row.id,
                    "display_name": row.display_name,
                    "enabled": row.enabled,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    async def set_model_enabled(self, model_id: str, enabled: bool):
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            row = await session.get(ModelRecord, model_id)
            if row is not None:
                row.enabled = enabled

    async def is_model_enabled(self, model_id: str) -> bool:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            row = await session.get(ModelRecord, model_id)
            if row is None:
                return True
            return row.enabled

    async def get_model_tags(self, model_id: str) -> Dict[str, str]:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            rows = await session.scalars(
                select(ModelTagRecord).where(ModelTagRecord.model_id == model_id)
            )
            return {row.key: row.value for row in rows}

    async def set_model_tags(self, model_id: str, tags: Dict[str, str]):
        """Полностью заменяет набор тегов модели."""
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            await session.execute(delete(ModelTagRecord).where(ModelTagRecord.model_id == model_id))
            for key, value in tags.items():
                if value:
                    session.add(ModelTagRecord(model_id=model_id, key=key, value=value))

    async def get_all_tags_by_models(self) -> Dict[str, Dict[str, str]]:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            rows = await session.scalars(select(ModelTagRecord))
            result: Dict[str, Dict[str, str]] = {}
            for row in rows:
                result.setdefault(row.model_id, {})[row.key] = row.value
            return result

    async def get_tag_types(self) -> List[Dict[str, str]]:
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory() as session:
            rows = await session.scalars(select(TagTypeRecord).order_by(TagTypeRecord.label))
            return [
                {
                    "key": row.key,
                    "label": row.label,
                    "description": row.description,
                }
                for row in rows
            ]

    async def add_tag_type(self, key: str, label: str, description: str = ""):
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            stmt = insert(TagTypeRecord).values(key=key, label=label, description=description)
            stmt = stmt.on_conflict_do_update(
                index_elements=[TagTypeRecord.key],
                set_={
                    "label": stmt.excluded.label,
                    "description": stmt.excluded.description,
                },
            )
            await session.execute(stmt)

    async def delete_tag_type(self, key: str):
        if self.session_factory is None:
            raise RuntimeError("Database is not connected")

        async with self.session_factory.begin() as session:
            await session.execute(delete(TagTypeRecord).where(TagTypeRecord.key == key))
