import uuid
from sqlalchemy import Column, String

from database import Base



class Pet(Base):
    __tablename__ = "pets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    pet_type = Column(String, nullable=False)
    affected_limb = Column(String, nullable=False)


class Converter(Base):
    __tablename__ = "converters"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    path_image = Column(String, nullable=True)
    path_video = Column(String, nullable=False)
    path_obj = Column(String, nullable=True)
    status = Column(String, nullable=False, default="processing")
