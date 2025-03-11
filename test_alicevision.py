#!/usr/bin/env python3
import os
import sys

from converter.services.alicevision_processor import AliceVisionProcessor


def main():
    # Verificar argumentos
    if len(sys.argv) < 3:
        print("Uso: python test_alicevision.py <diretório_imagens> <diretório_saída>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Definir caminho do AliceVision via argumento opcional ou variável de ambiente
    alicevision_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.environ.get(
            "ALICEVISION_BIN_PATH", "/home/pedro/dev/tcc/src/Framework/aliceVision/bin"
        )
    )

    print(f"Iniciando processamento com AliceVision:")
    print(f"- Diretório de entrada: {input_dir}")
    print(f"- Diretório de saída: {output_dir}")
    print(f"- Caminho do AliceVision: {alicevision_path}")

    # Inicializar e executar o processador
    try:
        processor = AliceVisionProcessor(
            input_directory=input_dir,
            output_directory=output_dir,
            alicevision_bin_path=alicevision_path,
        )
        processor.process_images()
        print("Processamento completo!")
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
