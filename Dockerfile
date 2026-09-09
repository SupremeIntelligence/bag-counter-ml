FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir openmim
RUN pip install --no-cache-dir mmengine==0.10.7
RUN mim install "mmcv==2.1.0"
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY models ./models
COPY rtmdet_tiny_8xb32-300e_coco.py .

RUN mkdir -p storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]