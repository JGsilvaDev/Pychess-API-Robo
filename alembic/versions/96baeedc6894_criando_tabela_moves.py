"""Criando tabela moves

Revision ID: 96baeedc6894
Revises: 37c9a310a1a0
Create Date: 2025-02-27 10:34:21.216776

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '96baeedc6894'
down_revision: Union[str, None] = '37c9a310a1a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Criando a tabela moves
    op.create_table(
        'moves',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('is_player', sa.Boolean, nullable=False),
        sa.Column('move', sa.String(4), nullable=False),
        sa.Column('board_string', sa.String(250), nullable=False),
        sa.Column('mv_quality', sa.String(10), nullable=True),
        sa.Column('game_id', sa.Integer, sa.ForeignKey('games.id', ondelete="CASCADE"), nullable=False),
        sa.Column('created_at', sa.String(255), nullable=False)
    )


def downgrade():
    # Removendo a tabela moves caso seja necessário reverter a migration
    op.drop_table('moves')
