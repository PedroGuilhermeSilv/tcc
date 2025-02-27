import os
from dataclasses import dataclass
from typing import Tuple, Optional
import subprocess

import cv2
from cv2.typing import MatLike
from pyexiv2 import Image as ExivImage

from object_values.type_videos import TypeVideos
from converter.services.alicevision_processor import AliceVisionProcessor


@dataclass
class VideoFrameExtractor:
    path_video: str
    path_image_metadata: str
    name_video: str
    format: TypeVideos
    output_3d_path: str = "src/tmp/3d_models"

    def execute(self) -> None:
        """Processa o vídeo, extraindo frames e gerando modelo 3D."""
        video_path = os.path.join(
            self.path_video, f"{self.name_video}.{self.format.value}"
        )
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo em: {video_path}")

        try:
            output_dir = self._process_video_frames(video)
            self._generate_3d_model(output_dir)
        finally:
            video.release()
            cv2.destroyAllWindows()

    def _process_video_frames(self, video: cv2.VideoCapture) -> str:
        output_dir = os.path.join("src/tmp", self.name_video)
        os.makedirs(output_dir, exist_ok=True)

        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break

            image_path = os.path.join(
                output_dir, f"{self.name_video}_{frame_count}.jpeg"
            )
            cv2.imwrite(image_path, frame)

            if not self._is_frame_blurry(frame, image_path):
                self._copy_image_metadata(self.path_image_metadata, image_path)

            frame_count += 1

        return output_dir

    def _is_frame_blurry(self, image: MatLike, image_path: str) -> bool:
        if image is None:
            print(f"Frame vazio detectado em: {image_path}")
            return True

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Pontuação de nitidez: {blur_score}")

        if blur_score <= 40:  # Imagem borrada
            os.remove(image_path)
            return True

        return False

    def _copy_image_metadata(self, source_path: str, destination_path: str) -> None:
        # Verificar se o arquivo de metadados existe
        if not os.path.exists(source_path) or os.path.isdir(source_path):
            print(
                f"Arquivo de metadados não encontrado: {source_path}. Pulando cópia de metadados."
            )
            return

        try:
            with ExivImage(source_path) as metadata:
                exif_data = metadata.read_exif()
            with ExivImage(destination_path) as output_metadata:
                output_metadata.modify_exif(exif_data)
        except Exception as e:
            print(f"Erro ao copiar metadados: {e}. Continuando sem metadados.")

    def _generate_3d_model(self, frames_directory: str) -> None:
        """Gera modelo 3D a partir dos frames extraídos."""
        output_dir = os.path.join(
            self.output_3d_path, os.path.basename(frames_directory)
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"Gerando modelo 3D em: {output_dir}")

        try:
            # Usar AliceVision diretamente
            from converter.services.alicevision_processor import AliceVisionProcessor

            # Procurar o AliceVision em vários locais possíveis
            possible_paths = [
                # 1. Caminho padrão do AliceVision
                "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/aliceVision/bin",
                # 2. Usar variável de ambiente
                os.environ.get("ALICEVISION_BIN_PATH", ""),
                # 3. Procurar no PATH
                "/usr/local/bin/aliceVision",
                "/usr/bin/aliceVision",
            ]

            alicevision_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    alicevision_path = path
                    break

            if alicevision_path:
                print(f"Usando AliceVision para processamento: {alicevision_path}")
                processor = AliceVisionProcessor(
                    input_directory=frames_directory,
                    output_directory=output_dir,
                    alicevision_bin_path=alicevision_path,
                    force_cpu=True,  # Opcional, remova se tiver GPU
                )
                processor.process_images()
                return

            # Se ainda não conseguiu, mostrar erro detalhado
            print("Erro: Não foi possível encontrar o AliceVision em nenhum local:")
            for path in possible_paths:
                print(f"- Tentado: {path}")
            print("Por favor, instale o AliceVision ou configure o caminho correto.")
            raise FileNotFoundError("AliceVision não encontrado")

        except Exception as e:
            print(f"Erro ao gerar modelo 3D: {e}")
            raise
