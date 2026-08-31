FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system rag && adduser --system --ingroup rag rag

COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/backend-requirements.txt

COPY --chown=rag:rag backend /app/backend

USER rag
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "backend.runtime_app:create_runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
