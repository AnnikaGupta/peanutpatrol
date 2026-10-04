FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# Cloud Run sets PORT; app.py reads it and binds 0.0.0.0.
ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "app.py"]
