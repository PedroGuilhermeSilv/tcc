import os
from dataclasses import dataclass
from converter.services.alicevision_processor import Processor
from converter.services.alicevision_processor import AliceVisionProcessor
import cv2
from cv2.typing import MatLike
from pyexiv2 import Image as ExivImage

from object_values.type_videos import TypeVideos


@dataclass
class VideoFrameExtractor:
    path_video: str
    path_image_metadata: str
    name_video: str
    format: TypeVideos
    output_3d_path: str = "src/tmp/3d_models"

    def execute(self) -> None:
        """Processa o vídeo, extraindo frames e gerando modelo 3D."""
        print(f"\n=== Iniciando processamento do vídeo: {self.name_video} ===")
        video_path = os.path.join(
            self.path_video, f"{self.name_video}.{self.format.value}"
        )
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo em: {video_path}")

        try:
            print("\n1. Extraindo frames do vídeo...")
            output_dir = self._process_video_frames(video)
            print("\n2. Iniciando geração do modelo 3D...")
            self._generate_3d_model(
                output_dir,
                AliceVisionProcessor(
                    input_directory=output_dir,
                    output_directory=self.output_3d_path,
                    alicevision_bin_path="/home/pedro/dev/tcc/src/Framework/aliceVision/bin",
                    force_cpu=True,
                ),
            )
            print("\n=== Processamento concluído com sucesso! ===")
        finally:
            video.release()
            cv2.destroyAllWindows()

    def _process_video_frames(self, video: cv2.VideoCapture) -> str:
        output_dir = os.path.join("src/tmp", self.name_video)
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        frames_processados = 0
        print(f"   → Diretório de saída: {output_dir}")

        while True:
            success, frame = video.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 10 == 0:  # Atualiza a cada 10 frames
                print(
                    f"   → Processando frame {frame_count} | Frames válidos: {frames_processados}"
                )

            image_path = os.path.join(
                output_dir, f"{self.name_video}_{frame_count}.jpeg"
            )
            cv2.imwrite(image_path, frame)

            if not self._is_frame_blurry(frame, image_path):
                frames_processados += 1

        print(f"\n   ✓ Concluído! Total de frames processados: {frame_count}")
        print(f"   ✓ Frames válidos mantidos: {frames_processados}")
        return output_dir

    def _is_frame_blurry(self, image: MatLike, image_path: str) -> bool:
        if image is None:
            print(f"   ⚠ Frame vazio detectado e descartado")
            return True

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score <= 40:
            return True

        return False

    def _copy_image_metadata(self, source_path: str, destination_path: str) -> None:
        if not os.path.exists(source_path) or os.path.isdir(source_path):
            print(f"   ⚠ Arquivo de metadados não encontrado: {source_path}")
            return

        try:
            with ExivImage(source_path) as metadata:
                exif_data = metadata.read_exif()
            with ExivImage(destination_path) as output_metadata:
                output_metadata.modify_exif(exif_data)
        except Exception as e:
            print(f"   ⚠ Erro ao copiar metadados: {str(e)}")

    def _generate_3d_model(self, frames_directory: str, processor: Processor) -> None:
        output_dir = os.path.join(
            self.output_3d_path, os.path.basename(frames_directory)
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"   → Diretório do modelo 3D: {output_dir}")

        try:
            processor.process_images()
            print("   ✓ Modelo 3D gerado com sucesso!")
            return

        except Exception as e:
            print(f"\n   ❌ Erro ao gerar modelo 3D: {str(e)}")
            raise
