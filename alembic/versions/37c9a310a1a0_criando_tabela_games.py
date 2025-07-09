"""Criando tabela games

Revision ID: 37c9a310a1a0
Revises: d822d6f9b670
Create Date: 2025-02-27 09:56:36.318828

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '37c9a310a1a0'
down_revision = 'd822d6f9b670'
branch_labels = None
depends_on = None


def upgrade():
    # Criando a tabela games
    op.create_table(
        'games',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.id'), nullable=False),
        sa.Column('player_win', sa.Boolean, nullable=False, default=False)
    )


def downgrade():
    # Removendo a tabela games caso seja necessário reverter a migration
    op.drop_table('games')
