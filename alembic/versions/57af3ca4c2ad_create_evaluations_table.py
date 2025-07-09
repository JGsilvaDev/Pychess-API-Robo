"""create evaluations table

Revision ID: 57af3ca4c2ad
Revises: 96baeedc6894
Create Date: 2025-04-16 16:07:01.827271

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '57af3ca4c2ad'
down_revision: Union[str, None] = '96baeedc6894'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        'evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('game_id', sa.Integer(), nullable=True),
        sa.Column('evaluation', sa.Integer(), nullable=True),
        sa.Column('depth', sa.Integer(), nullable=True),
        sa.Column('win_probability_white', sa.Float(), nullable=True),
        sa.Column('win_probability_black', sa.Float(), nullable=True),
        sa.Column('last_updated', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['game_id'], ['games.id']),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
   op.drop_table('evaluations')
