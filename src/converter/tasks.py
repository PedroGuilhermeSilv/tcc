from celery_app import celery
from database import get_db
from sqlalchemy.orm import Session
from converter.services.export_img import VideoFrameExtractor
from object_values.type_videos import TypeVideos
from converter.services.video_service import update_video_status


@celery.task(name="converter.tasks.process_video")
def process_video(video_id, video_path, pet_name):
    """Processa um vídeo para extrair frames e gerar modelo 3D"""
    print(f"Processando vídeo: {video_id}, {video_path}, {pet_name}")
    try:

        extractor = VideoFrameExtractor(
            path_video=video_path,
            path_image_metadata="src/tmp/base-sem-flash.HEIC",
            name_video=pet_name,
            format=TypeVideos.MOV,
        )
        extractor.execute()
        # Atualiza o status no banco de dados para "finalizado"
        db: Session = next(get_db())
        update_video_status(db, video_id, "finalizado")
        print(f"Vídeo processado com sucesso: {video_id}")
    except Exception as e:
        print(f"Erro ao processar vídeo {video_id}: {e}")
        db: Session = next(get_db())
        update_video_status(db, video_id, "error")
        raise
