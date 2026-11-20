# docker build --target serve -t fraudshield-serve .     (api, consumer, producer)
# docker build --target train -t fraudshield-train .     (pipeline, retrain, drift jobs)
# docker build --target dashboard -t fraudshield-dashboard .
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/app/data ARTIFACTS_DIR=/app/artifacts FEAST_REPO=/app/feature_repo
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

FROM base AS serve
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml params.yaml ./
COPY src ./src
COPY feature_repo ./feature_repo
COPY feature_repo/feature_store.docker.yaml ./feature_repo/feature_store.yaml
RUN pip install -e . --no-deps && useradd -m app && mkdir -p data artifacts reports && chown -R app /app
USER app
ENV FEATURE_BACKEND=feast
EXPOSE 8000
CMD ["uvicorn", "fraudshield.api:create_default", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS train
COPY requirements-serve.txt requirements-train.txt ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-train.txt \
 && pip install torch_geometric
COPY pyproject.toml params.yaml Makefile ./
COPY src ./src
COPY scripts ./scripts
COPY feature_repo ./feature_repo
COPY feature_repo/feature_store.docker.yaml ./feature_repo/feature_store.yaml
RUN pip install -e . --no-deps && mkdir -p data artifacts reports
ENV FEATURE_BACKEND=feast
CMD ["sh", "scripts/run_pipeline.sh"]

FROM python:3.11-slim AS dashboard
WORKDIR /app
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt
COPY dashboard ./dashboard
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
