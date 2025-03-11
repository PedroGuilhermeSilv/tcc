import os
import subprocess
import requests
import json
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path

from abc import ABC, abstractmethod


class Processor(ABC):
    input_directory: Union[str, Path]
    output_directory: Union[str, Path]

    @abstractmethod
    def process_images(self) -> None:
        pass


@dataclass
class AliceVisionProcessor(Processor):
    input_directory: Union[str, Path]
    output_directory: Union[str, Path]
    alicevision_bin_path: Union[str, Path] = os.environ.get(
        "ALICEVISION_BIN_PATH",
        "/home/pedro/dev/tcc/tcc/src/Framework/aliceVision/bin",
    )
    force_cpu: bool = True
    verbose: bool = True

    def __post_init__(self):
        # Converte caminhos para Path caso sejam strings e garante o caminho absoluto
        self.input_directory = Path(self.input_directory).absolute()
        self.output_directory = Path(self.output_directory).absolute()
        self.alicevision_bin_path = Path(self.alicevision_bin_path).absolute()

        # Verificar se o diretório dos binários existe
        if not self.alicevision_bin_path.exists():
            raise ValueError(
                f"O diretório de binários AliceVision não existe: {self.alicevision_bin_path}"
            )

        # Configurar diretórios usando Path
        self.alicevision_root = self.alicevision_bin_path.parent.parent
        self.alicevision_lib_path = self.alicevision_bin_path.parent / "lib"

        # Configurar diretório share e arquivo OCIO usando Path
        self.alicevision_share = self.alicevision_root / "share" / "aliceVision"
        self.alicevision_share.mkdir(parents=True, exist_ok=True)

        self.ocio_path = self.alicevision_share / "config.ocio"
        if not self.ocio_path.exists():
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
            file_path = self.alicevision_share / file_name
            if not file_path.exists():
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
            print("\n=== Iniciando pipeline de reconstrução 3D ===")
            # Criar diretório de cache
            cache_dir = Path(self.output_directory) / "cache"
            cache_dir.mkdir(exist_ok=True)
            # Pipeline completo do AliceVision
            steps = [
                (
                    "Extração de características",
                    lambda: self._run_feature_extraction(cache_dir),
                ),
                ("Matching de imagens", lambda: self._run_image_matching(cache_dir)),
                (
                    "Matching de características",
                    lambda: self._run_feature_matching(cache_dir),
                ),
                (
                    "Reconstrução da estrutura",
                    lambda: self._run_structure_from_motion(cache_dir),
                ),
                (
                    "Preparação da cena densa",
                    lambda: self._run_prepare_dense_scene(cache_dir),
                ),
                (
                    "Estimativa de mapas de profundidade",
                    lambda: self._run_depth_map_estimation(cache_dir),
                ),
                (
                    "Filtragem de mapas de profundidade",
                    lambda: self._run_depth_map_filter(cache_dir),
                ),
                ("Geração da malha", lambda: self._run_meshing(cache_dir)),
                ("Filtragem da malha", lambda: self._run_mesh_filtering(cache_dir)),
                ("Texturização", lambda: self._run_texturing(cache_dir)),
            ]

            total_steps = len(steps)
            for i, (step_name, step_func) in enumerate(steps, 1):
                print(f"\n[{i}/{total_steps}] {step_name}")
                print("=" * (len(step_name) + 8))
                step_func()
                print(f"✓ {step_name} concluído")

            print("\n=== Reconstrução 3D completada com sucesso! ===")
            print(f"Modelo final salvo em: {self.output_directory}")

        except Exception as e:
            print("\n❌ Erro durante o processamento AliceVision:")
            print(f"   {str(e)}")
            raise

    def _get_bin_path(self, binary_name: str) -> str:
        """Retorna o caminho completo para um binário do AliceVision"""
        return str(Path(self.alicevision_bin_path) / binary_name)

    def _run_command(self, cmd: List[str], env: Optional[dict] = None) -> None:
        """Executa um comando do AliceVision com o ambiente configurado"""
        print("\n📋 Executando comando:")
        print(f"   {' '.join(cmd)}")
        # Configurar ambiente
        current_env = os.environ.copy()
        if env:
            current_env.update(env)
        # Forçar uso de CPU se necessário
        if self.force_cpu:
            current_env["CUDA_VISIBLE_DEVICES"] = "-1"
            current_env["ALICEVISION_USE_CUDA"] = "0"
        # Configurar caminho das bibliotecas
        lib_paths = [
            str(self.alicevision_lib_path),
            "/usr/lib",
            "/usr/local/lib",
            "/home/pedro/dev/tcc/src/Framework/lib",
            "/home/pedro/dev/tcc/src/Framework/aliceVision/lib",
        ]
        current_env["LD_LIBRARY_PATH"] = (
            ":".join([path for path in lib_paths if os.path.exists(path)])
            + ":"
            + current_env.get("LD_LIBRARY_PATH", "")
        )
        # Configurar variáveis de ambiente do AliceVision
        current_env["ALICEVISION_ROOT"] = str(self.alicevision_root)
        current_env["ALICEVISION_SHARE"] = str(self.alicevision_share)
        current_env["OCIO"] = str(self.ocio_path)
        print(f"Com LD_LIBRARY_PATH: {current_env['LD_LIBRARY_PATH']}")
        print(f"Com ALICEVISION_ROOT: {current_env['ALICEVISION_ROOT']}")
        print(f"Com ALICEVISION_SHARE: {current_env['ALICEVISION_SHARE']}")
        print(f"Com OCIO: {current_env['OCIO']}")
        try:
            result = subprocess.run(
                cmd,
                env=current_env,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print("\n📝 Saída do comando:")
                for line in result.stdout.split("\n"):
                    if line.strip():
                        print(f"   {line}")
        except subprocess.CalledProcessError as e:
            print("\n❌ Erro na execução do comando:")
            if e.stdout:
                print("\n📝 Saída padrão:")
                for line in e.stdout.split("\n"):
                    if line.strip():
                        print(f"   {line}")
            if e.stderr:
                print("\n⚠️ Saída de erro:")
                for line in e.stderr.split("\n"):
                    if line.strip():
                        print(f"   {line}")
            raise

    def _run_feature_extraction(self, cache_dir: Path) -> None:
        """Extrai características das imagens"""
        # Criar diretórios necessários
        features_dir = cache_dir / "features"
        features_dir.mkdir(exist_ok=True)
        # Criar arquivo de lista de imagens
        images_list = cache_dir / "images.txt"
        with open(images_list, "w") as f:
            for img in sorted(os.listdir(str(self.input_directory))):
                if img.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = str(Path(self.input_directory) / img)
                    f.write(f"{img_path}\n")
        # Verificar se há imagens
        if not images_list.stat().st_size:
            raise ValueError(
                f"Nenhuma imagem encontrada no diretório: {self.input_directory}"
            )
        print(f"Processando {sum(1 for _ in open(images_list))} imagens")
        # Gerar arquivo SfM a partir da lista de imagens
        sfm_file = cache_dir / "sfm.json"
        sensor_db = self.alicevision_share / "cameraSensors.db"
        if not sensor_db.exists():
            print(
                f"Aviso: Arquivo de banco de dados de sensores não encontrado: {sensor_db}"
            )
            print("Tentando prosseguir sem o banco de dados de sensores...")
            sensor_db = None
        cmd = [
            self._get_bin_path("aliceVision_cameraInit"),
            "--imageFolder",
            str(self.input_directory),
            "--defaultFieldOfView",
            "45",
            "--verboseLevel",
            "info",
            "--output",
            str(sfm_file),
        ]
        if sensor_db:
            cmd.extend(["--sensorDatabase", str(sensor_db)])
        else:
            cmd.extend(["--defaultCameraModel", "pinhole"])
        self._run_command(cmd)
        # Verificar se o arquivo SfM foi gerado
        if not sfm_file.exists():
            raise RuntimeError(f"Falha ao gerar arquivo SfM: {sfm_file}")
        # Verificar integridade do arquivo
        with open(sfm_file, "r") as f:
            if not f.read().strip():
                raise RuntimeError(f"Arquivo SfM vazio: {sfm_file}")
        # Configurar comando de extração de características
        cmd = [
            self._get_bin_path("aliceVision_featureExtraction"),
            "--input",
            str(sfm_file),
            "--output",
            str(features_dir),
            "--describerTypes",
            "sift",
            "--verboseLevel",
            "info",
        ]
        if self.force_cpu:
            cmd.extend(["--forceCpuExtraction", "1"])
        self._run_command(cmd)

    def _run_image_matching(self, cache_dir: Path) -> None:
        """Realiza matching entre imagens"""
        # Criar diretórios necessários
        matches_dir = cache_dir / "matches"
        matches_dir.mkdir(exist_ok=True)
        # Verificar se o arquivo SfM existe
        sfm_file = cache_dir / "sfm.json"
        if not sfm_file.exists():
            raise RuntimeError(f"Arquivo SfM não encontrado: {sfm_file}")
        # Debug: Imprimir conteúdo do sfm.json
        print("\nConteúdo do sfm.json:")
        with open(sfm_file) as f:
            sfm_data = json.load(f)
            print(json.dumps(sfm_data, indent=2))
        # Verificar se o arquivo tree existe
        tree_file = self.alicevision_share / "vlfeat_K80L3.SIFT.tree"
        if not tree_file.exists():
            raise RuntimeError(f"Arquivo tree não encontrado: {tree_file}")
        # Gerar pares de imagens
        image_pairs_file = matches_dir / "image_pairs.txt"
        views = sfm_data.get("views", [])
        pairs = []
        for i in range(len(views) - 1):
            view_id1 = views[i].get("viewId")
            view_id2 = views[i + 1].get("viewId")
            if view_id1 is not None and view_id2 is not None:
                try:
                    view_id1 = int(view_id1)
                    view_id2 = int(view_id2)
                except ValueError:
                    print(
                        f"Aviso: ViewIds '{view_id1}' e/ou '{view_id2}' não são números inteiros. Usando índices {i} e {i+1}."
                    )
                    view_id1 = i
                    view_id2 = i + 1
                pairs.append(f"{view_id1} {view_id2}")
        print("\nPares de imagens gerados:")
        for pair in pairs:
            print(f"  - {pair}")
        with open(image_pairs_file, "w") as f:
            f.write("\n".join(pairs))
            f.write("\n")  # Adiciona uma linha em branco no final

    def _run_feature_matching(self, cache_dir: Path) -> None:
        """Executa o matching de características"""
        image_pairs_file = cache_dir / "matches" / "image_pairs.txt"
        if not image_pairs_file.exists():
            raise RuntimeError(
                f"Arquivo image_pairs.txt não encontrado em: {image_pairs_file}. Verifique o comando aliceVision_imageMatching."
            )
        # Debug: Imprimir conteúdo do arquivo de pares
        print("\nConteúdo do arquivo image_pairs.txt:")
        with open(image_pairs_file) as f:
            print(f.read())
        # Verificar se o diretório de features existe e tem arquivos
        features_dir = cache_dir / "features"
        if not features_dir.exists() or not any(features_dir.iterdir()):
            raise RuntimeError(
                f"Diretório de features vazio ou não encontrado: {features_dir}"
            )
        cmd = [
            self._get_bin_path("aliceVision_featureMatching"),
            "--input",
            str(cache_dir / "sfm.json"),
            "--output",
            str(cache_dir / "matches"),
            "--featuresFolders",
            str(features_dir),
            "--imagePairs",
            str(image_pairs_file),
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
        try:
            self._run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"\nErro no feature matching. Saída do comando:")
            if hasattr(e, "stdout"):
                print("Saída padrão:")
                print(e.stdout)
            if hasattr(e, "stderr"):
                print("Saída de erro:")
                print(e.stderr)
            raise

    def _run_structure_from_motion(self, cache_dir: Path) -> None:
        """Executa a reconstrução da estrutura a partir do movimento"""
        env = os.environ.copy()
        # Configurar ambiente para CPU/GPU
        if self.force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env["ALICEVISION_USE_CUDA"] = "0"
        # Criar diretório sfm se não existir
        sfm_dir = cache_dir / "sfm"
        sfm_dir.mkdir(exist_ok=True)
        # Usar o arquivo sfm.json diretamente do cache_dir como entrada
        input_sfm = cache_dir / "sfm.json"
        output_sfm = sfm_dir / "sfm.json"
        if not input_sfm.exists():
            raise FileNotFoundError(
                f"Arquivo de entrada sfm.json não encontrado: {input_sfm}"
            )
        cmd = [
            self._get_bin_path("aliceVision_incrementalSfM"),
            "--input",
            str(input_sfm),
            "--output",
            str(output_sfm),
            "--matchesFolder",
            str(cache_dir / "matches"),
            "--featuresFolders",
            str(cache_dir / "features"),
            "--minAngleInitialPair",
            "3",
            "--maxAngleInitialPair",
            "50",
            "--minNumberOfMatches",
            "50",
        ]
        self._run_command(cmd, env=env)
        # Verificar se o arquivo foi gerado
        if not output_sfm.exists():
            # Se o arquivo não foi gerado no diretório sfm, verificar no cache_dir
            if not input_sfm.exists():
                raise FileNotFoundError(
                    f"O arquivo sfm.json não foi gerado em nenhum local esperado:\n"
                    f"- {output_sfm}\n"
                    f"- {input_sfm}"
                )
            else:
                # Se o arquivo existe apenas no cache_dir, copiar para o diretório sfm
                import shutil

                shutil.copy2(str(input_sfm), str(output_sfm))
                print(f"Arquivo sfm.json copiado de {input_sfm} para {output_sfm}")
        # Debug: mostrar conteúdo do arquivo gerado
        try:
            with open(output_sfm, "r") as f:
                sfm_data = json.load(f)
                print(
                    f"\nConteúdo do arquivo sfm.json gerado:\n{json.dumps(sfm_data, indent=2)}"
                )
        except Exception as e:
            print(f"Aviso: Não foi possível ler o arquivo sfm.json para debug: {e}")

    def _run_prepare_dense_scene(self, cache_dir: Path) -> None:
        """Prepara a cena para a reconstrução densa"""
        # Adicionado: verificar o conteúdo do diretório cache
        print("Conteúdo do diretório cache:")
        for item in cache_dir.iterdir():
            print(f"  - {item}")

        cmd = [
            self._get_bin_path("aliceVision_prepareDenseScene"),
            "--input",
            str(cache_dir / "sfm" / "sfm.json"),
            "--output",
            str(cache_dir / "mvsData"),
        ]
        self._run_command(cmd)

    def _run_depth_map_estimation(self, cache_dir: Path) -> None:
        """Estima mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapEstimation"),
            "--input",
            str(cache_dir / "mvsData" / "sfm.json"),
            "--output",
            str(cache_dir / "depthMap"),
        ]
        self._run_command(cmd)

    def _run_depth_map_filter(self, cache_dir: Path) -> None:
        """Filtra mapas de profundidade"""
        cmd = [
            self._get_bin_path("aliceVision_depthMapFiltering"),
            "--input",
            str(cache_dir / "mvsData" / "sfm.json"),
            "--depthMapsFolder",
            str(cache_dir / "depthMap"),
            "--output",
            str(cache_dir / "depthMap_filtered"),
        ]
        self._run_command(cmd)

    def _run_meshing(self, cache_dir: Path) -> None:
        """Cria a malha 3D"""
        cmd = [
            self._get_bin_path("aliceVision_meshing"),
            "--input",
            str(cache_dir / "mvsData" / "sfm.json"),
            "--depthMapsFolder",
            str(cache_dir / "depthMap_filtered"),
            "--output",
            str(cache_dir / "mesh.obj"),
        ]
        self._run_command(cmd)

    def _run_mesh_filtering(self, cache_dir: Path) -> None:
        """Filtra a malha 3D"""
        cmd = [
            self._get_bin_path("aliceVision_meshFiltering"),
            "--input",
            str(cache_dir / "mesh.obj"),
            "--output",
            str(cache_dir / "mesh_filtered.obj"),
        ]
        self._run_command(cmd)

    def _run_texturing(self, cache_dir: Path) -> None:
        """Aplica textura à malha 3D"""
        output_path = str(Path(self.output_directory) / "textured_model")
        cmd = [
            self._get_bin_path("aliceVision_texturing"),
            "--input",
            str(cache_dir / "mesh_filtered.obj"),
            "--imagesFolder",
            str(self.input_directory),
            "--inputMesh",
            str(cache_dir / "mesh_filtered.obj"),
            "--output",
            output_path,
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
