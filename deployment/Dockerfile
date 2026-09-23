FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    opencv-python-headless==4.12.0.88 pandas numpy

COPY foosball_core.py main.py ./
COPY calibration_artifacts/ ./calibration_artifacts/
COPY model_weights/ ./model_weights/

ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 300
