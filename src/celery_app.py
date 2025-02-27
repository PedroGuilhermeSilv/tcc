from celery import Celery
import os

# Configuração direta em vez de importar app
celery = Celery(
    "converter",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

# Configurações adicionais se necessário
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

# Inclua os módulos que contêm tarefas
celery.autodiscover_tasks(["converter"])
