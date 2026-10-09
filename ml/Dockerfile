FROM python:3.11.9-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Build context is ml/. For a full run, mount the entire Git checkout at
# /workspace/repository so exports can archive actual repository source bytes.
WORKDIR /workspace/repository/ml
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-export.txt -r requirements-data.txt -r requirements-dev.txt
COPY . .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scripts/run_training_pipeline.py"]
CMD ["--help"]
