# Short-lived test runner: executes pytest against a read-only workspace mount.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "pytest==8.3.5" \
    && useradd --create-home --uid 10001 runner

USER runner
WORKDIR /workspace

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
