import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AliceVisionProcessor:
    input_directory: str
    output_directory: str
    alicevision_bin_path: str = os.environ.get(
        "ALICEVISION_BIN_PATH",
        "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/aliceVision/bin",
    )

    def __post_init__(self):
        # Converte caminhos relativos para absolutos
        self.input_directory = os.path.abspath(self.input_directory)
        self.output_directory = os.path.abspath(self.output_directory)

        # Verificar se o diretório dos binários existe
        if not os.path.exists(self.alicevision_bin_path):
            raise ValueError(
                f"O diretório de binários AliceVision não existe: {self.alicevision_bin_path}"
                "\nConfigure a variável de ambiente ALICEVISION_BIN_PATH corretamente."
            )

        self.alicevision_bin_path = os.path.abspath(self.alicevision_bin_path)

        # Procurar diretório de bibliotecas em vários locais possíveis
        possible_lib_paths = [
            # 1. Dentro do diretório AliceVision
            os.path.join(os.path.dirname(self.alicevision_bin_path), "lib"),
            # 2. No mesmo nível que o diretório bin do AliceVision
            os.path.join(
                os.path.dirname(os.path.dirname(self.alicevision_bin_path)), "lib"
            ),
            # 3. Dois níveis acima (instalação típica)
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(self.alicevision_bin_path))
                ),
                "lib",
            ),
            # 4. No próprio diretório bin (algumas distribuições colocam tudo junto)
            self.alicevision_bin_path,
        ]

        # Testar cada caminho possível
        self.alicevision_lib_path = None
        for lib_path in possible_lib_paths:
            if os.path.exists(lib_path):
                # Verificar se contém as bibliotecas do AliceVision
                try:
                    files = os.listdir(lib_path)
                    if any(f.startswith("libaliceVision") for f in files):
                        self.alicevision_lib_path = lib_path
                        print(f"Encontradas bibliotecas do AliceVision em: {lib_path}")
                        break
                except Exception:
                    pass

        # Se não encontrar, usar o diretório bin como fallback
        if not self.alicevision_lib_path:
            print(
                "Aviso: Não foi possível encontrar as bibliotecas do AliceVision. Usando o diretório bin como fallback."
            )
            self.alicevision_lib_path = self.alicevision_bin_path

        print(f"Diretório de entrada: {self.input_directory}")
        print(f"Diretório de saída: {self.output_directory}")
        print(f"Caminho do AliceVision: {self.alicevision_bin_path}")
        print(f"Caminho de bibliotecas do AliceVision: {self.alicevision_lib_path}")

        # Verificar a instalação
        self._verify_installation()

    def process_images(self) -> None:
        """Processa imagens usando AliceVision para criar modelo 3D"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        # Cria diretórios para cada etapa do pipeline
        cache_dir = os.path.join(self.output_directory, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Usando AliceVision em: {self.alicevision_bin_path}")

        # Executa o pipeline do AliceVision
        try:
            # 1. Feature Extraction
            self._run_feature_extraction(cache_dir)

            # 2. Image Matching
            self._run_image_matching(cache_dir)

            # 3. Feature Matching
            self._run_feature_matching(cache_dir)

            # 4. Structure from Motion
            self._run_structure_from_motion(cache_dir)

            # 5. Prepare Dense Scene
            self._run_prepare_dense_scene(cache_dir)

            # 6. Depth Map Estimation
            self._run_depth_map_estimation(cache_dir)

            # 7. Depth Map Filter
            self._run_depth_map_filter(cache_dir)

            # 8. Meshing
            self._run_meshing(cache_dir)

            # 9. Mesh Filtering
            self._run_mesh_filtering(cache_dir)

            # 10. Texturing
            self._run_texturing(cache_dir)

            print(
                f"3D reconstruction completed. Output saved to {self.output_directory}"
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during AliceVision processing: {e}")
            raise

    def _get_bin_path(self, binary_name: str) -> str:
        """Retorna o caminho completo para um binário do AliceVision"""
        return os.path.join(self.alicevision_bin_path, binary_name)

    def _run_command(self, cmd: List[str]) -> None:
        """Executa um comando com o ambiente configurado para as bibliotecas do AliceVision"""
        # Copiar o ambiente atual
        env = os.environ.copy()

        # Adicionar ou atualizar o LD_LIBRARY_PATH
        current_ld_lib_path = env.get("LD_LIBRARY_PATH", "")
        if current_ld_lib_path:
            env["LD_LIBRARY_PATH"] = (
                f"{self.alicevision_lib_path}:{current_ld_lib_path}"
            )
        else:
            env["LD_LIBRARY_PATH"] = self.alicevision_lib_path or ""

        # Configurar ALICEVISION_ROOT - normalmente é o diretório pai do diretório bin
        alicevision_root = os.path.dirname(os.path.dirname(self.alicevision_bin_path))
        env["ALICEVISION_ROOT"] = alicevision_root

        print(f"Executando: {' '.join(cmd)}")
        print(f"Com LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
        print(f"Com ALICEVISION_ROOT: {env['ALICEVISION_ROOT']}")

        # Executar o comando com o ambiente modificado
        subprocess.run(cmd, check=True, env=env)

    def _run_feature_extraction(self, cache_dir: str) -> None:
        """Executa a extração de características das imagens"""
        cmd = [
            self._get_bin_path("aliceVision_featureExtraction"),
            "--input",
            self.input_directory,
            "--output",
            os.path.join(cache_dir, "features"),
            "--describerTypes",
            "sift",
            "--forceCpuExtraction",
            "1",
        ]
        self._run_command(cmd)

    def _run_image_matching(self, cache_dir: str) -> None:
        """Executa o matching de imagens"""
        cmd = [
            self._get_bin_path("aliceVision_imageMatching"),
            "--input",
            os.path.join(cache_dir, "features", "sfm.json"),
            "--output",
            os.path.join(cache_dir, "matches"),
            "--tree",
            os.path.join(cache_dir, "features", "tree.json"),
        ]
        self._run_command(cmd)

    def _run_feature_matching(self, cache_dir: str) -> None:
        """Executa o matching de características"""
        cmd = [
            self._get_bin_path("aliceVision_featureMatching"),
            "--input",
            os.path.join(cache_dir, "features", "sfm.json"),
            "--output",
            os.path.join(cache_dir, "matches"),
            "--imagePairs",
            os.path.join(cache_dir, "matches", "image_pairs.txt"),
        ]
        self._run_command(cmd)

    def _run_structure_from_motion(self, cache_dir: str) -> None:
        """Executa a reconstrução da estrutura a partir do movimento"""
        cmd = [
            self._get_bin_path("aliceVision_incrementalSfM"),
            "--input",
            os.path.join(cache_dir, "features", "sfm.json"),
            "--output",
            os.path.join(cache_dir, "sfm"),
            "--matchesFolder",
            os.path.join(cache_dir, "matches"),
        ]
        self._run_command(cmd)

    def _run_prepare_dense_scene(self, cache_dir: str) -> None:
        """Prepara a cena para reconstrução densa"""
        cmd = [
            self._get_bin_path("aliceVision_prepareDenseScene"),
            "--input",
            os.path.join(cache_dir, "sfm", "sfm.abc"),
            "--output",
            os.path.join(cache_dir, "mvsData"),
        ]
        self._run_command(cmd)

    def _run_depth_map_estimation(self, cache_dir: str) -> None:
        """Estima mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapEstimation"),
            "--input",
            os.path.join(cache_dir, "mvsData", "sfm.abc"),
            "--output",
            os.path.join(cache_dir, "depthMap"),
        ]
        self._run_command(cmd)

    def _run_depth_map_filter(self, cache_dir: str) -> None:
        """Filtra mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapFiltering"),
            "--input",
            os.path.join(cache_dir, "mvsData", "sfm.abc"),
            "--depthMapsFolder",
            os.path.join(cache_dir, "depthMap"),
            "--output",
            os.path.join(cache_dir, "depthMap_filtered"),
        ]
        self._run_command(cmd)

    def _run_meshing(self, cache_dir: str) -> None:
        """Cria a malha 3D"""
        cmd = [
            self._get_bin_path("aliceVision_meshing"),
            "--input",
            os.path.join(cache_dir, "mvsData", "sfm.abc"),
            "--depthMapsFolder",
            os.path.join(cache_dir, "depthMap_filtered"),
            "--output",
            os.path.join(cache_dir, "mesh.obj"),
        ]
        self._run_command(cmd)

    def _run_mesh_filtering(self, cache_dir: str) -> None:
        """Filtra a malha 3D"""
        cmd = [
            self._get_bin_path("aliceVision_meshFiltering"),
            "--input",
            os.path.join(cache_dir, "mesh.obj"),
            "--output",
            os.path.join(cache_dir, "mesh_filtered.obj"),
        ]
        self._run_command(cmd)

    def _run_texturing(self, cache_dir: str) -> None:
        """Aplica textura à malha 3D"""
        cmd = [
            self._get_bin_path("aliceVision_texturing"),
            "--input",
            os.path.join(cache_dir, "mesh_filtered.obj"),
            "--imagesFolder",
            self.input_directory,
            "--inputMesh",
            os.path.join(cache_dir, "mesh_filtered.obj"),
            "--output",
            os.path.join(self.output_directory, "textured_model"),
        ]
        self._run_command(cmd)

    def _verify_installation(self) -> None:
        """Verifica se o AliceVision está corretamente instalado e configurado"""
        # Verificar se os binários existem
        feature_extraction_bin = self._get_bin_path("aliceVision_featureExtraction")
        if not os.path.exists(feature_extraction_bin):
            raise ValueError(
                f"Binário AliceVision não encontrado: {feature_extraction_bin}"
            )

        # Verificar permissões
        if not os.access(feature_extraction_bin, os.X_OK):
            print(
                f"Aviso: Binário {feature_extraction_bin} não tem permissão de execução."
            )
            print("Tentando corrigir permissões...")
            try:
                os.chmod(feature_extraction_bin, 0o755)  # rwxr-xr-x
                print("Permissões corrigidas.")
            except Exception as e:
                print(f"Erro ao corrigir permissões: {e}")

        # Listar bibliotecas no diretório lib para debug
        print("Bibliotecas disponíveis no diretório lib:")
        try:
            for file in sorted(os.listdir(self.alicevision_lib_path)):
                if file.endswith(".so") or ".so." in file:
                    print(f"  - {file}")
        except Exception as e:
            print(f"Erro ao listar bibliotecas: {e}")
