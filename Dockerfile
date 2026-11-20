FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY backend/app ./app
COPY --from=frontend /web/dist ./static
RUN useradd --create-home acoustic && chown -R acoustic:acoustic /app
USER acoustic
ENV ACOUSTICSPACE_FRONTEND_DIR=/app/static
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
