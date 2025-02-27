from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from pydantic import BaseModel
from typing import List, Optional
import os
from uuid import uuid4
from sqlalchemy.orm import Session

from converter.services.export_img import VideoFrameExtractor
from object_values.type_videos import TypeVideos
from celery_app import celery
from database import get_db
from converter.services.video_service import update_video_status, create_pet
from converter.model.entity import Pet, Converter
from converter.tasks import process_video

router = APIRouter(prefix="/api/videos")


class VideoUploadRequest(BaseModel):
    pet_name: str
    pet_type: str
    affected_limb: str


class VideoResponse(BaseModel):
    id: str
    status: str
    progress: float
    model_3d_url: Optional[str] = None


@router.post("/", response_model=VideoResponse)
async def upload_video(
    pet_name: str = Form(...),
    pet_type: str = Form(...),
    affected_limb: str = Form(...),
    video_file: UploadFile = File(...),
):
    video_id = str(uuid4())
    video_path = f"src/tmp/uploads/{video_id}"
    os.makedirs(video_path, exist_ok=True)

    file_path = os.path.join(video_path, f"{pet_name}.mp4")
    with open(file_path, "wb") as buffer:
        content = await video_file.read()
        buffer.write(content)

    # Cria um novo pet no banco de dados usando o modelo SQLAlchemy
    pet = Pet(name=pet_name, pet_type=pet_type, affected_limb=affected_limb)

    # Também cria um registro na tabela converters
    converter = Converter(
        id=video_id, name=pet_name, path_video=file_path, status="processing"
    )

    db: Session = next(get_db())
    create_pet(db, pet)
    db.add(converter)
    db.commit()

    # Envia a tarefa para a fila usando o novo caminho
    process_video.delay(video_id, video_path, pet_name)

    return VideoResponse(id=video_id, status="processing", progress=0)


@router.get("/{video_id}/status", response_model=VideoResponse)
async def get_video_status(video_id: str):
    return VideoResponse(
        id=video_id,
        status="processing",
        progress=50,
    )


@router.get("/", response_model=List[VideoResponse])
async def list_videos():
    # Implement listing of all video processing jobs
    # This should query your database
    return []
