from sqlalchemy import Column, Integer, Boolean, String, ForeignKey, DateTime
from database.database import Base
from datetime import datetime

class RobotToken(Base):
    __tablename__ = 'robot_tokens'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    token = Column(String, unique=True, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


