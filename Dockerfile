# CSST106 Computer Vision Labs: container image (any Docker host).
# Build & run locally:  docker build -t cv-labs . && docker run -p 8501:8501 cv-labs
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/home/user/app/models/torch \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    YOLO_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download the DeepLabV3 weights at build time so the first visitor does not wait
RUN python -c "from cv_core.lab7_segmentation import load_deeplab; load_deeplab()"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
