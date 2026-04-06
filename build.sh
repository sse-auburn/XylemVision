#!/bin/bash
# Build and run XylemVision.
#
# Usage:
#   bash build.sh           → GPU build, runs in foreground
#   bash build.sh -d        → GPU build, runs in background (detached)
#   bash build.sh --cpu     → CPU-only build, runs in foreground
#   bash build.sh --cpu -d  → CPU-only build, runs in background

DETACH=0
CPU=0

for arg in "$@"; do
  case $arg in
    -d)    DETACH=1 ;;
    --cpu) CPU=1 ;;
  esac
done

if [ "$CPU" = "1" ]; then
  DOCKERFILE="Dockerfile.cpu"
  IMAGE_NAME="xylemvision-cpu"
  GPU_FLAG=""
  echo "==> Mode: CPU-only"
else
  DOCKERFILE="Dockerfile"
  IMAGE_NAME="xylemvision"
  GPU_FLAG="--gpus all"
  echo "==> Mode: GPU"
fi

echo "==> Stopping and removing old container (xylemvision-app)..."
docker stop xylemvision-app 2>/dev/null
docker rm   xylemvision-app 2>/dev/null

echo "==> Removing old image ($IMAGE_NAME)..."
docker rmi $IMAGE_NAME 2>/dev/null

echo "==> Pruning build cache and dangling images..."
docker builder prune -f
docker image prune -f

echo "==> Building fresh image from $DOCKERFILE..."
docker build -f $DOCKERFILE -t $IMAGE_NAME . || { echo "BUILD FAILED"; exit 1; }

echo "==> Starting container..."
if [ "$DETACH" = "1" ]; then
  docker run $GPU_FLAG \
    -p 8000:8000 \
    --name xylemvision-app \
    -d \
    $IMAGE_NAME
  echo "==> Running at http://localhost:8000"
  echo "==> View logs: docker logs -f xylemvision-app"
else
  docker run $GPU_FLAG \
    -p 8000:8000 \
    --name xylemvision-app \
    $IMAGE_NAME
fi
