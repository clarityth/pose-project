import datetime
from datetime import timezone, timedelta

from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy import func
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password = Column(String)

    profile = relationship("UserProfile", back_populates="user", uselist=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    gender = Column(String)
    age = Column(Integer)
    weight = Column(Integer)
    height = Column(Integer)

    user = relationship("User", back_populates="profile")


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    from sqlalchemy import func
    created_at = Column(DateTime(timezone=True), server_default=func.timezone('Asia/Seoul', func.now()), nullable=False)
    user_profile_id = Column(Integer, ForeignKey("user_profiles.id"))
    image_filename = Column(String)
    shoulder_height_diff_px = Column(Float)
    hip_height_diff_px = Column(Float)
    torso_vertical_tilt_deg = Column(Float)
    ear_hip_vertical_tilt_deg = Column(Float)
    shoulder_line_horizontal_tilt_deg = Column(Float)
    hip_line_horizontal_tilt_deg = Column(Float)

    user_profile = relationship("UserProfile", backref="features")
