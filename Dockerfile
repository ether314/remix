# syntax=docker/dockerfile:1
FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, from the PyTorch index, so nothing later pulls the multi-GB CUDA build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchaudio==2.5.1

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Small additions live in their own layer so they never reinstall the heavy ones above.
COPY requirements-extra.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-extra.txt

COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests
RUN pip install --no-deps -e .

# Model weights download on first use into /models (a named volume in compose.yaml).
ENV REMIX_DATA=/data \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /data
ENTRYPOINT ["remix"]
