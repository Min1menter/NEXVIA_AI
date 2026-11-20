# Build from the repo root:  docker build -f services/inference/Dockerfile .
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY services/common /app/common
COPY services/inference /app/inference
RUN useradd -r -u 10001 app && mkdir -p /results && chown app /results
USER app
CMD ["python", "-m", "inference.main"]
