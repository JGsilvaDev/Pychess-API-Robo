from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from datetime import datetime
from database.database import Base

class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), unique=True)
    evaluation = Column(Integer)  # Centipawns
    depth = Column(Integer)
    win_probability_white = Column(Float)
    win_probability_black = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)
