FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install deps first for better layer caching
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .    
# includes app.py, migrator.py, mcp_openapi/...
COPY migrations/ /app/migrations/
COPY seeds/ /app/seeds/
ENV PYTHONPATH=/app

# Fail fast if the package is missing
# quick sanity
RUN python - <<'PY'
import psycopg, fastapi, uvicorn
print("imports ok")
PY

USER 10001:10001
EXPOSE 8080
CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
