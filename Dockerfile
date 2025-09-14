FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install deps first for better layer caching
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source so 'mcp_openapi' is available
COPY src/ . 
# Optional: ensure Python sees /app as a package root
ENV PYTHONPATH=/app

# Fail fast if the package is missing
RUN python -c "import uvicorn, mcp_openapi; print('import ok')"

USER 10001:10001
EXPOSE 8080
CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8080"]
