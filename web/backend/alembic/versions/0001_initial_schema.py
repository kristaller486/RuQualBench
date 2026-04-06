"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-01 17:10:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


tag_types_table = sa.table(
    "tag_types",
    sa.column("key", sa.String()),
    sa.column("label", sa.String()),
    sa.column("description", sa.Text()),
)


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.String(length=128), primary_key=True, nullable=False),
        sa.Column("display_name", sa.String(length=512), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_table(
        "tag_types",
        sa.Column("key", sa.String(length=128), primary_key=True, nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
    )

    op.create_table(
        "model_tags",
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.ForeignKeyConstraint(["key"], ["tag_types.key"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("model_id", "key"),
    )

    op.bulk_insert(
        tag_types_table,
        [
            {"key": "hf_url", "label": "HuggingFace", "description": "Ссылка на модель на HuggingFace"},
            {"key": "license", "label": "Лицензия", "description": "Лицензия модели"},
            {"key": "api", "label": "API", "description": "API, через которое тестировалась модель"},
            {"key": "engine", "label": "Движок", "description": "Версия движка инференса"},
            {"key": "params", "label": "Параметры", "description": "Количество параметров модели"},
            {"key": "notes", "label": "Заметки", "description": "Дополнительные заметки"},
        ],
    )


def downgrade() -> None:
    op.drop_table("model_tags")
    op.drop_table("tag_types")
    op.drop_table("models")
