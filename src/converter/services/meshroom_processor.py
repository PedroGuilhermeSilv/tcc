import os
import subprocess
from dataclasses import dataclass


@dataclass
class MeshroomProcessor:
    input_directory: str
    output_directory: str
    meshroom_binary: str = os.environ.get(
        "MESHROOM_BINARY", "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/meshroom_batch"
    )
    force_cpu: bool = False
    verbose: bool = True

    def __post_init__(self):
        # Converte caminhos relativos para absolutos
        self.input_directory = os.path.abspath(self.input_directory)
        self.output_directory = os.path.abspath(self.output_directory)
        self.meshroom_binary = os.path.abspath(self.meshroom_binary)

        if not os.path.exists(self.meshroom_binary):
            raise ValueError(f"Binário Meshroom não encontrado: {self.meshroom_binary}")

        # Verificar permissões
        if not os.access(self.meshroom_binary, os.X_OK):
            print(f"Corrigindo permissões para: {self.meshroom_binary}")
            os.chmod(self.meshroom_binary, 0o755)

        print(f"Diretório de entrada: {self.input_directory}")
        print(f"Diretório de saída: {self.output_directory}")
        print(f"Caminho do Meshroom: {self.meshroom_binary}")

    def process_images(self) -> None:
        """Processa imagens usando Meshroom para criar modelo 3D"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Configurar comando base
        command = [
            self.meshroom_binary,
            "--input",
            self.input_directory,
            "--output",
            self.output_directory,
            "--cache",
            os.path.join(self.output_directory, "cache"),
            "--save",
            os.path.join(self.output_directory, "pipeline.mg"),
        ]

        # Adicionar opções
        if self.force_cpu:
            command.extend(["--forceCpu"])
        if self.verbose:
            command.extend(
                ["-v", "info"]
            )  # Níveis: fatal, error, warning, info, debug, trace

        try:
            print(f"Executando: {' '.join(command)}")

            # Configurar ambiente
            env = os.environ.copy()
            env["PATH"] = (
                f"{os.path.dirname(self.meshroom_binary)}:{env.get('PATH', '')}"
            )

            # Executar comando
            result = subprocess.run(
                command, check=True, env=env, capture_output=True, text=True
            )

            if result.stdout:
                print("Saída do Meshroom:")
                print(result.stdout)

            print(
                f"Reconstrução 3D completada. Saída salva em: {self.output_directory}"
            )

        except subprocess.CalledProcessError as e:
            print(f"Erro durante o processamento Meshroom:")
            print(f"Código de retorno: {e.returncode}")
            if e.stdout:
                print("Saída padrão:")
                print(e.stdout)
            if e.stderr:
                print("Saída de erro:")
                print(e.stderr)
            raise
