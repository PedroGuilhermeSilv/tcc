import os
import subprocess
import requests
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
    force_cpu: bool = False
    verbose: bool = True

    def __post_init__(self):
        # Converte caminhos relativos para absolutos
        self.input_directory = os.path.abspath(self.input_directory)
        self.output_directory = os.path.abspath(self.output_directory)
        self.alicevision_bin_path = os.path.abspath(self.alicevision_bin_path)

        # Verificar se o diretório dos binários existe
        if not os.path.exists(self.alicevision_bin_path):
            raise ValueError(
                f"O diretório de binários AliceVision não existe: {self.alicevision_bin_path}"
            )

        # Configurar diretórios
        self.alicevision_root = os.path.dirname(
            os.path.dirname(self.alicevision_bin_path)
        )
        self.alicevision_lib_path = os.path.join(
            os.path.dirname(self.alicevision_bin_path), "lib"
        )

        # Configurar diretório share e arquivo OCIO
        self.alicevision_share = os.path.join(
            self.alicevision_root, "share/aliceVision"
        )
        os.makedirs(self.alicevision_share, exist_ok=True)

        self.ocio_path = os.path.join(self.alicevision_share, "config.ocio")
        if not os.path.exists(self.ocio_path):
            with open(self.ocio_path, "w") as f:
                f.write(
                    """ocio_profile_version: 2

search_path: ""
strictparsing: true
luma: [0.2126, 0.7152, 0.0722]

roles:
  default: raw
  scene_linear: raw

displays:
  sRGB:
    - !<View> {name: Raw, colorspace: raw}

colorspaces:
  - !<ColorSpace>
    name: raw
    family: raw
    equalitygroup: ""
    bitdepth: 32f
    isdata: false
    allocation: uniform"""
                )

        # Verificar se as bibliotecas existem
        if not os.path.exists(self.alicevision_lib_path):
            raise ValueError(
                f"Diretório de bibliotecas não encontrado: {self.alicevision_lib_path}"
            )

        # Listar bibliotecas disponíveis para debug
        print("Bibliotecas disponíveis no diretório lib:")
        for lib in sorted(os.listdir(self.alicevision_lib_path)):
            print(f"  - {lib}")

        print(f"Usando AliceVision em: {self.alicevision_bin_path}")
        print(f"Diretório de entrada: {self.input_directory}")
        print(f"Diretório de saída: {self.output_directory}")
        print(f"Arquivo OCIO: {self.ocio_path}")

        # Verificar e criar arquivos essenciais
        essential_files = {
            "cameraSensors.db": "https://github.com/alicevision/AliceVision/raw/develop/src/aliceVision/sensorDB/cameraSensors.db",
            "vlfeat_K80L3.SIFT.tree": "https://github.com/alicevision/AliceVision/raw/develop/src/aliceVision/voctree/vlfeat_K80L3.SIFT.tree",
        }

        for file_name, url in essential_files.items():
            file_path = os.path.join(self.alicevision_share, file_name)
            if not os.path.exists(file_path):
                print(f"Baixando arquivo essencial: {file_name}")
                try:
                    response = requests.get(url, timeout=10)
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(f"Arquivo {file_name} baixado com sucesso.")
                except Exception as download_error:
                    print(f"Falha ao baixar {file_name}: {download_error}")
                    print("O processo pode não funcionar corretamente.")

    def process_images(self) -> None:
        """Processa imagens usando AliceVision para criar modelo 3D"""
        try:
            # Criar diretório de cache
            cache_dir = os.path.join(self.output_directory, "cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Pipeline completo do AliceVision
            self._run_feature_extraction(cache_dir)
            self._run_image_matching(cache_dir)
            self._run_feature_matching(cache_dir)
            self._run_structure_from_motion(cache_dir)
            self._run_prepare_dense_scene(cache_dir)
            self._run_depth_map_estimation(cache_dir)
            self._run_depth_map_filter(cache_dir)
            self._run_meshing(cache_dir)
            self._run_mesh_filtering(cache_dir)
            self._run_texturing(cache_dir)

            print(
                f"Reconstrução 3D completada. Saída salva em: {self.output_directory}"
            )

        except Exception as e:
            print(f"Erro durante o processamento AliceVision: {e}")
            raise

    def _get_bin_path(self, binary_name: str) -> str:
        """Retorna o caminho completo para um binário do AliceVision"""
        return os.path.join(self.alicevision_bin_path, binary_name)

    def _run_command(self, cmd: List[str]) -> None:
        """Executa um comando do AliceVision com o ambiente configurado"""
        print(f"Executando: {' '.join(cmd)}")

        # Configurar ambiente
        env = os.environ.copy()

        # Configurar caminho das bibliotecas
        lib_paths = [
            self.alicevision_lib_path,
            "/usr/lib",
            "/usr/local/lib",
            "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/lib",
            "/home/pedro/dev/tcc/src/Meshroom-2023.3.0/aliceVision/lib",
        ]
        env["LD_LIBRARY_PATH"] = (
            ":".join([path for path in lib_paths if os.path.exists(path)])
            + ":"
            + env.get("LD_LIBRARY_PATH", "")
        )

        # Configurar variáveis de ambiente do AliceVision
        env["ALICEVISION_ROOT"] = self.alicevision_root
        env["ALICEVISION_SHARE"] = self.alicevision_share
        env["OCIO"] = self.ocio_path

        print(f"Com LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
        print(f"Com ALICEVISION_ROOT: {env['ALICEVISION_ROOT']}")
        print(f"Com ALICEVISION_SHARE: {env['ALICEVISION_SHARE']}")
        print(f"Com OCIO: {env['OCIO']}")

        try:
            result = subprocess.run(
                cmd, check=True, env=env, capture_output=True, text=True
            )

            if result.stdout and self.verbose:
                print("Saída:")
                print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar {cmd[0]}:")
            if e.stdout:
                print("Saída padrão:")
                print(e.stdout)
            if e.stderr:
                print("Saída de erro:")
                print(e.stderr)
            raise

    def _run_feature_extraction(self, cache_dir: str) -> None:
        """Extrai características das imagens"""
        # Criar diretórios necessários
        features_dir = os.path.join(cache_dir, "features")
        os.makedirs(features_dir, exist_ok=True)

        # Criar arquivo de lista de imagens
        images_list = os.path.join(cache_dir, "images.txt")
        with open(images_list, "w") as f:
            for img in sorted(os.listdir(self.input_directory)):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(self.input_directory, img)
                    f.write(f"{img_path}\n")

        # Verificar se há imagens
        if not os.path.getsize(images_list):
            raise ValueError(
                f"Nenhuma imagem encontrada no diretório: {self.input_directory}"
            )

        print(f"Processando {sum(1 for _ in open(images_list))} imagens")

        # Gerar arquivo SfM a partir da lista de imagens
        sfm_file = os.path.join(cache_dir, "sfm.json")

        sensor_db = os.path.join(self.alicevision_share, "cameraSensors.db")
        if not os.path.exists(sensor_db):
            print(
                f"Aviso: Arquivo de banco de dados de sensores não encontrado: {sensor_db}"
            )
            print("Tentando prosseguir sem o banco de dados de sensores...")
            sensor_db = None

        cmd = [
            self._get_bin_path("aliceVision_cameraInit"),
            "--imageFolder",
            self.input_directory,
            "--defaultFieldOfView",
            "45",
            "--verboseLevel",
            "info",
            "--output",
            sfm_file,
        ]

        if sensor_db:
            cmd.extend(["--sensorDatabase", sensor_db])
        else:
            cmd.extend(["--defaultCameraModel", "pinhole"])

        self._run_command(cmd)

        # Verificar se o arquivo SfM foi gerado
        if not os.path.exists(sfm_file):
            raise RuntimeError(f"Falha ao gerar arquivo SfM: {sfm_file}")

        # Verificar integridade do arquivo
        with open(sfm_file, "r") as f:
            if not f.read().strip():
                raise RuntimeError(f"Arquivo SfM vazio: {sfm_file}")

        # Configurar comando de extração de características
        cmd = [
            self._get_bin_path("aliceVision_featureExtraction"),
            "--input",
            sfm_file,
            "--output",
            features_dir,
            "--describerTypes",
            "sift",
            "--verboseLevel",
            "info",
        ]

        if self.force_cpu:
            cmd.extend(["--forceCpuExtraction", "1"])

        self._run_command(cmd)

    def _run_image_matching(self, cache_dir: str) -> None:
        """Realiza matching entre imagens"""
        # Criar diretórios necessários
        matches_dir = os.path.join(cache_dir, "matches")
        os.makedirs(matches_dir, exist_ok=True)

        cmd = [
            self._get_bin_path("aliceVision_imageMatching"),
            "--input",
            os.path.join(cache_dir, "sfm.json"),
            "--output",
            matches_dir,
            "--featuresFolders",
            os.path.join(cache_dir, "features"),
            "--method",
            "VocabularyTree",
            "--tree",
            os.path.join(self.alicevision_share, "vlfeat_K80L3.SIFT.tree"),
            "--verboseLevel",
            "info",
        ]
        self._run_command(cmd)

    def _run_feature_matching(self, cache_dir: str) -> None:
        """Executa o matching de características"""
        image_pairs_file = os.path.join(cache_dir, "matches", "image_pairs.txt")
        if not os.path.exists(image_pairs_file):
            raise RuntimeError(f"Arquivo image_pairs.txt não encontrado em: {image_pairs_file}. Verifique o comando aliceVision_imageMatching.")
        cmd = [
            self._get_bin_path("aliceVision_featureMatching"),
            "--input",
            os.path.join(cache_dir, "sfm.json"),  # Caminho correto
            "--output",
            os.path.join(cache_dir, "matches"),
            "--featuresFolders",
            os.path.join(cache_dir, "features"),
            "--imagePairs",
            os.path.join(cache_dir, "matches", "image_pairs.txt"),
            "--describerTypes",
            "sift",
            "--photometricMatchingMethod",
            "ANN_L2",
            "--geometricEstimator",
            "acransac",
            "--geometricFilterType",
            "fundamental_matrix",
            "--distanceRatio",
            "0.8",
            "--verboseLevel",
            "info",
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
            os.path.join(cache_dir, "sfm", "sfm.json"),
            "--output",
            os.path.join(cache_dir, "mvsData"),
        ]
        self._run_command(cmd)

    def _run_depth_map_estimation(self, cache_dir: str) -> None:
        """Estima mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapEstimation"),
            "--input",
            os.path.join(cache_dir, "mvsData", "sfm.json"),
            "--output",
            os.path.join(cache_dir, "depthMap"),
        ]
        self._run_command(cmd)

    def _run_depth_map_filter(self, cache_dir: str) -> None:
        """Filtra mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapFiltering"),
            "--input",
            os.path.join(cache_dir, "mvsData", "sfm.json"),
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
            os.path.join(cache_dir, "mvsData", "sfm.json"),
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
