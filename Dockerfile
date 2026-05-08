FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies + Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    build-essential \
    libpq-dev \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Create necessary directories for weights
RUN mkdir -p /app/weight/SAM /app/weight/YOLO

# Download model weights
RUN pip install gdown && \
    gdown https://drive.google.com/uc?id=16QARfz1cpumYtwBSf23nlBWtr3hweTQy -O /app/weight/SAM/sam_vit_l_0b3195.pth && \
    gdown https://drive.google.com/uc?id=1maEVUeXS3wCywabZNeO9R-TsBDNanzAS -O /app/weight/YOLO/best.pt

# Copy project files
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8000

# Run with gunicorn
CMD ["gunicorn", "root.wsgi:application", "--bind", "0.0.0.0:8000", "--chdir", "/app", "--timeout", "180", "--graceful-timeout", "30", "--workers", "1", "--threads", "4"]
