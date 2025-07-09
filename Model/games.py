from sqlalchemy import Column, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database.database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    player_win = Column(Integer, default=False)

    user = relationship("User")  # Relacionamento opcional
