# One image for both services; docker-compose.yml picks the command. Build context rules are in .dockerignore.
FROM python:3.13-slim

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv==0.12.7

WORKDIR /app

# The lock file first, so a source or README change does not reinstall every dependency (CUDA PyTorch is about an hour)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY README.md ./
COPY src ./src
COPY configs ./configs
# The UI theme: Streamlit reads .streamlit/config.toml from the folder it starts in, /app
COPY .streamlit ./.streamlit
RUN uv sync --frozen

EXPOSE 8000 8501
CMD ["/app/.venv/bin/uvicorn", "citerag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
