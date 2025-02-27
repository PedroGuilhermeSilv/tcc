import os
from pathlib import Path
import shutil


def setup_environment():
    # Obtém o caminho absoluto do diretório raiz do projeto
    project_root = Path(__file__).parent.absolute()

    # Define os caminhos do AliceVision
    alicevision_root = project_root / "src" / "Meshroom-2023.3.0"
    alicevision_share = alicevision_root / "share" / "aliceVision"
    alicevision_lib = alicevision_root / "lib"
    alicevision_bin = alicevision_root / "aliceVision" / "bin"

    # Verifica e corrige permissões dos arquivos essenciais
    essential_files = [
        alicevision_share / "vlfeat_K80L3.SIFT.tree",
        alicevision_share / "cameraSensors.db",
        alicevision_share / "config.ocio",
    ]

    for file in essential_files:
        if file.exists():
            # Garante permissões de leitura para o arquivo
            os.chmod(file, 0o644)
            print(f"✅ Permissões ajustadas para: {file}")
        else:
            print(f"❌ Arquivo não encontrado: {file}")

    # Garante permissões de execução para binários
    if alicevision_bin.exists():
        for binary in alicevision_bin.glob("*"):
            if binary.is_file():
                os.chmod(binary, 0o755)
                print(f"✅ Permissões de execução ajustadas para: {binary}")

    # Configura as variáveis de ambiente
    os.environ["ALICEVISION_ROOT"] = str(alicevision_root)
    os.environ["ALICEVISION_SHARE"] = str(alicevision_share)

    # Atualiza LD_LIBRARY_PATH
    lib_paths = [
        str(alicevision_lib),
        "/usr/lib",
        "/usr/local/lib",
        str(alicevision_root / "lib"),
        str(alicevision_root / "aliceVision" / "lib"),
    ]

    if "LD_LIBRARY_PATH" in os.environ:
        lib_paths.append(os.environ["LD_LIBRARY_PATH"])

    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_paths)

    # Configura OCIO
    os.environ["OCIO"] = str(alicevision_share / "config.ocio")

    return {
        "ALICEVISION_ROOT": os.environ["ALICEVISION_ROOT"],
        "ALICEVISION_SHARE": os.environ["ALICEVISION_SHARE"],
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "OCIO": os.environ["OCIO"],
    }


if __name__ == "__main__":
    env_vars = setup_environment()
    print("\nVariáveis de ambiente configuradas:")
    for key, value in env_vars.items():
        print(f"{key}={value}")
