import os
import subprocess
from pathlib import Path


def check_file_permissions(file_path):
    if not os.access(file_path, os.R_OK):
        return False, "sem permissão de leitura"
    return True, "OK"


def check_alicevision_dependencies():
    # Verifica se o Meshroom está instalado no local correto
    meshroom_path = Path("src/Meshroom-2023.3.0")
    if not meshroom_path.exists():
        print("❌ Meshroom não encontrado em src/Meshroom-2023.3.0")
        return False

    # Verifica arquivos essenciais
    essential_files = [
        "share/aliceVision/vlfeat_K80L3.SIFT.tree",
        "share/aliceVision/cameraSensors.db",
        "share/aliceVision/config.ocio",
        "aliceVision/bin/aliceVision_cameraInit",
        "aliceVision/bin/aliceVision_featureExtraction",
        "aliceVision/bin/aliceVision_imageMatching",
    ]

    missing_files = []
    for file in essential_files:
        file_path = meshroom_path / file
        if not file_path.exists():
            missing_files.append(file)
            print(f"❌ Arquivo não encontrado: {file}")
        else:
            can_read, status = check_file_permissions(file_path)
            if can_read:
                print(f"✅ Arquivo encontrado: {file} ({status})")

                # Verifica tamanho do arquivo tree
                if "vlfeat_K80L3.SIFT.tree" in str(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   📊 Tamanho do arquivo tree: {size/1024/1024:.2f} MB")
            else:
                print(f"⚠️  Arquivo encontrado mas {status}: {file}")
                missing_files.append(f"{file} ({status})")

    # Verifica se OpenCV está instalado
    try:
        import cv2

        print("✅ OpenCV está instalado")
    except ImportError:
        print("❌ OpenCV não está instalado")
        missing_files.append("opencv-python")

    # Verifica variáveis de ambiente
    env_vars = ["LD_LIBRARY_PATH", "ALICEVISION_ROOT", "ALICEVISION_SHARE"]

    print("\nVariáveis de ambiente:")
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"❌ {var} não definida")
            missing_files.append(f"Variável de ambiente: {var}")

    if missing_files:
        print("\nArquivos/pacotes/variáveis faltando:")
        for file in missing_files:
            print(f"- {file}")
        return False

    print("\n✅ Todas as dependências estão instaladas!")
    return True


if __name__ == "__main__":
    check_alicevision_dependencies()
