import psycopg2
import os
from urllib.parse import urlparse

# Conectar ao banco de dados
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/converter"
)

# Usar urllib.parse para extrair componentes da URL corretamente
url = urlparse(DATABASE_URL)
db_name = url.path[1:]  # Remove a barra inicial
user = url.username
password = url.password
host = url.hostname
port = url.port or 5432  # Porta padrão se não especificada

# Conectar ao banco de dados
conn = psycopg2.connect(
    dbname=db_name, user=user, password=password, host=host, port=port
)

# Criar um cursor
cur = conn.cursor()

# Executar comandos SQL para criar tabelas
cur.execute(
    """
CREATE TABLE IF NOT EXISTS pets (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    pet_type VARCHAR(255) NOT NULL,
    affected_limb VARCHAR(255) NOT NULL
);
"""
)

cur.execute(
    """
CREATE TABLE IF NOT EXISTS converters (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path_image VARCHAR(255),
    path_video VARCHAR(255) NOT NULL,
    path_obj VARCHAR(255),
    status VARCHAR(255) NOT NULL DEFAULT 'processing'
);
"""
)

# Confirmar as alterações
conn.commit()

# Fechar a conexão
cur.close()
conn.close()

print("Tabelas criadas com sucesso!")
