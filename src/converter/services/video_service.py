from sqlalchemy.orm import Session
from converter.model.entity import Converter, Pet  # Importe ambos do mesmo arquivo
from uuid import UUID


def create_pet(db: Session, pet: Pet):
    db.add(pet)
    db.commit()
    db.refresh(pet)
    return pet


def update_video_status(db: Session, video_id: UUID, status: str):
    video = db.query(Converter).filter_by(id=video_id).first()
    if video:
        video.status = status
        db.commit()
        db.refresh(video)
    return video
