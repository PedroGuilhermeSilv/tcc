import os
import subprocess
from dataclasses import dataclass

@dataclass
class MeshroomProcessor:
    input_directory: str
    output_directory: str
    meshroom_binary: str = os.environ.get(
        "MESHROOM_BINARY",
        "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/meshroom_batch"
    )

    def __post_init__(self):
        # Converte caminhos relativos para absolutos
        self.input_directory = os.path.abspath(self.input_directory)
        self.output_directory = os.path.abspath(self.output_directory)
        self.meshroom_binary = os.path.abspath(self.meshroom_binary)

        if not os.path.exists(self.meshroom_binary):
            raise ValueError(f"Binário Meshroom não encontrado: {self.meshroom_binary}")

        print(f"Diretório de entrada: {self.input_directory}")
        print(f"Diretório de saída: {self.output_directory}")
        print(f"Caminho do Meshroom: {self.meshroom_binary}")

    def process_images(self) -> None:
        """Processa imagens usando Meshroom para criar modelo 3D"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        command = [
            self.meshroom_binary,
            "--input",
            self.input_directory,
            "--output",
            self.output_directory,
        ]

        try:
            print(f"Executando: {' '.join(command)}")
            subprocess.run(command, check=True)
            print(f"3D reconstruction completed. Output saved to {self.output_directory}")
        except subprocess.CalledProcessError as e:
            print(f"Error during Meshroom processing: {e}")
            raise 