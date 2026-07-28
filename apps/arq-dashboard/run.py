from __future__ import annotations

import os

import uvicorn
from arq.connections import RedisSettings
from arq_dashboard.core import settings
from arq_dashboard.main import create_app


redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
queue_name = os.getenv("ARQ_QUEUE_NAME", "repomind:indexing")

settings.ARQ_QUEUES[queue_name] = RedisSettings(host=redis_host, port=redis_port)

uvicorn.run(
    create_app(),
    host="0.0.0.0",
    port=8000,
    log_level="info",
)
