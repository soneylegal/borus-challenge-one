# ==========================================
# Multi-stage Build para Borus
# ==========================================

# Estágio 1: Builder / Dependências
FROM python:3.11-slim AS builder

WORKDIR /build

# Instala ferramentas essenciais para compilação se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de definição de pacote
COPY pyproject.toml .

# Cria virtualenv e instala dependências de produção
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# ==========================================
# Estágio 2: Runtime enxuto e seguro
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Variáveis de ambiente padrão do Python e aplicação
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHROMA_PERSIST_DIRECTORY="/app/data/chroma" \
    DOCS_SOURCE_DIR="/app/docs_source" \
    APP_HOST="0.0.0.0" \
    APP_PORT="8000"

# Instala curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root por segurança
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copia virtualenv compilado do estágio builder
COPY --from=builder /opt/venv /opt/venv

# Cria pastas de volume com permissão adequada
RUN mkdir -p /app/data/chroma /app/docs_source && \
    chmod -R 777 /app/data && \
    chown -R appuser:appgroup /app

# Copia código-fonte e documentos de exemplo
COPY --chown=appuser:appgroup app /app/app
COPY --chown=appuser:appgroup docs_source /app/docs_source
COPY --chown=appuser:appgroup README.md /app/README.md

USER appuser

EXPOSE 8000

# Healthcheck interno do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
