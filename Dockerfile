FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Slim, deterministic deps
RUN pip install --no-cache-dir fastapi==0.114.2 uvicorn==0.30.6 httpx==0.27.2

COPY src/app.py .

# Non-root
USER 10001:10001
EXPOSE 8080

CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
