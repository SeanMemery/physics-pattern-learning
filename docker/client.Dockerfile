FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/SeanMemery/pattern-learning" \
      org.opencontainers.image.description="Pattern Learning execution client"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STACK_ROOT="/opt/pattern-learning-stack" \
    PATH="/opt/pattern-client-venv/bin:${PATH}"

WORKDIR /src

COPY README.md requirements.txt /src/
COPY src /src/src
COPY configs /src/configs
COPY data /src/data
COPY scripts /src/scripts
COPY deps /src/deps
COPY docker/docker-entrypoint.sh /usr/local/bin/pattern-client-entrypoint

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && mkdir -p "${STACK_ROOT}" /app/state /app/artifacts \
 && cp -R /src/. "${STACK_ROOT}" \
 && python -m venv /opt/pattern-client-venv \
 && /opt/pattern-client-venv/bin/pip install --no-cache-dir -r "${STACK_ROOT}/requirements.txt" \
 && cd "${STACK_ROOT}" \
 && ln -s "${STACK_ROOT}" /app/project \
 && ln -s "${STACK_ROOT}/venv" /app/venv \
 && rm -rf "${STACK_ROOT}/build" \
 && chmod +x /usr/local/bin/pattern-client-entrypoint

RUN cd "${STACK_ROOT}" \
 && /opt/pattern-client-venv/bin/python -m tooling.setup_envs learning phyre iphyre kinetix pooltool \
 && rm -rf "${STACK_ROOT}/build"

WORKDIR /app

ENTRYPOINT ["/usr/local/bin/pattern-client-entrypoint"]
