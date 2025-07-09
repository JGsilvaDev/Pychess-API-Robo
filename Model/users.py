from sqlalchemy import Column, Integer, String
from database.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), nullable=False, unique=True)
    password = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    total_games = Column(Integer, default=0)
    rating = Column(Integer, default=0)
