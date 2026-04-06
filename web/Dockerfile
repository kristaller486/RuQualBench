# Stage 1: Сборка фронтенда
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python-runtime
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости бэкенда
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Код бэкенда
COPY backend/ ./web/backend/

# Собранный фронтенд
COPY --from=frontend /app/frontend/dist ./web/frontend/dist

# Пустой __init__.py для пакета web
RUN touch ./web/__init__.py

EXPOSE 8080

CMD ["sh", "-c", "alembic -c web/backend/alembic.ini upgrade head && uvicorn web.backend.main:app --host 0.0.0.0 --port 8080"]
