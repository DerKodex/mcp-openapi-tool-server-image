FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY src/app.py .
COPY src/requirements.txt .

# Slim, deterministic deps
RUN pip install --no-cache-dir -r requirements.txt

# Non-root
USER 10001:10001
EXPOSE 8080

CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
