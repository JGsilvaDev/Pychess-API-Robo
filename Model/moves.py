from sqlalchemy import Column, Integer, Boolean, String, ForeignKey
from sqlalchemy.orm import relationship
from database.database import Base

class Move(Base):
    __tablename__ = "moves"

    id = Column(Integer, primary_key=True, autoincrement=True)
    is_player = Column(Boolean, nullable=False)
    move = Column(String(4), nullable=False)
    board_string = Column(String(250), nullable=False)
    mv_quality = Column(String(10), nullable=True)
    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(String(250), nullable=False)

    game = relationship("Game", backref="moves")  # Relacionamento opcional
