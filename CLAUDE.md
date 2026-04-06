# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This App Does

XylemVision is a Django web app for automated quantitative analysis of plant root anatomy from microscope cross-section images. It uses a two-stage ML pipeline (YOLO v8 + SAM) to detect and segment xylem vessels, vascular bundles, and total root area, then exports measurements to Excel.

## Running the App

### Docker (GPU) — primary deployment method
```bash
# Build and run in detached mode
bash build.sh -d

# CPU-only build
bash build.sh --cpu -d
```

The GPU Dockerfile expects model weights to be pre-downloaded locally:
- `weight/SAM/sam_vit_l_0b3195.pth` (~1.2 GB)
- `weight/YOLO/best.pt` (~120 MB)

The CPU Dockerfile downloads them automatically via `gdown` during build.

App runs at **http://localhost:8000**

### Hot-updating without rebuilding
```bash
# Copy any changed file into the running container, then reload gunicorn
docker cp <local-file> xylemvision-app:/app/<file>
docker exec xylemvision-app kill -HUP 1
```

On Linux (no Docker Desktop), prefix docker commands with `DOCKER_HOST=unix:///var/run/docker.sock` if the default socket is not found.

### Running locally (development)
```bash
pip install -r requirements.txt
python manage.py runserver
```

No database migrations are needed — the app uses SQLite and has no models.

## Architecture

### Request Flow
```
Browser upload → /analyze_stream/ (SSE) → engine.progressive_yolo_sam()
                                               ↓
                                       1. YOLO detection
                                       2. Hierarchy post-processing (4 rules)
                                       3. SAM segmentation (hierarchical)
                                       4. Mask refinement + metric calculation
                                               ↓
                                      SSE events → frontend canvas (Fabric.js)
```

### Key Files
| File | Role |
|------|------|
| `analysis/engine.py` | ML pipeline: YOLO inference, 4 hierarchy rules, SAM segmentation, metric calculation |
| `analysis/views.py` | 9 API endpoints; SSE streaming, SAM prompting, reanalysis, Excel/ZIP export |
| `analysis/utils.py` | Image processing: mask blending, contour extraction, SAM prompt generation |
| `analysis/configs.py` | Model paths, device selection, hyperparameters (`YOLO_CONF=0.55`, `ALPHA=0.65`, etc.) |
| `analysis/templates/upload.html` | Single-page UI (~59KB): Fabric.js canvas, XHR upload + SSE stream handler, annotation tools |
| `root/settings.py` | Django settings; `DEBUG=True`, `ALLOWED_HOSTS=['*']`, single `analysis` app |

### ML Pipeline Details (`engine.py`)
- **4 Hierarchy Rules**: Filter anatomically impossible YOLO detections (e.g. VB ≥ root, xylem ≥ VB). Rule 4 recovers a missing root detection using Otsu thresholding on the VB-masked image.
- **Hierarchical SAM**: Runs SAM on xylems first → blacks them out → runs on VBs → blacks out both → runs on root. Prevents mask overlap.
- **Coordinate scaling**: Images >2048px are resized for inference; scale factor is tracked and coordinates are mapped back to original resolution for interactive annotations and exports.

### Upload & Streaming
- The browser uses `XMLHttpRequest` (not `fetch`) for `/analyze_stream/` so that `xhr.upload.onprogress` can show real-time upload percentage and bytes transferred before analysis begins.
- Django buffers all uploaded files to disk before the view function runs — the "upload" phase in the UI reflects this real transfer time, not GPU wait time.
- `DATA_UPLOAD_MAX_NUMBER_FILES = 1000` in `settings.py` (Django default is 100 — batches over 100 images will silently fail without this).

### Caching
Results are stored in an in-memory `_last_analysis_cache` dict keyed by filename. Lost on container restart. Interactive reanalysis (`/reanalyze/`) reads from and writes back to this cache.

### API Endpoints (`analysis/urls.py`)
- `GET /` — main UI
- `POST /analyze_stream/` — SSE batch analysis
- `POST /sam_prompt/` — interactive SAM box/point prompt
- `POST /reanalyze/` — recalculate metrics from edited polygons
- `POST /merge_masks/` — merge polygon selections
- `GET /download_xlsx/<filename>/` — single-image Excel export
- `POST /download_all_xlsx/` — batch Excel export
- `POST /export_training/` — CVAT-XML + image as ZIP (single)
- `POST /export_training_batch/` — CVAT-XML + images as ZIP (batch)
- `POST /download_overlays/` — segmentation overlay images as ZIP

## Model Weights

Weights are not in the repo. For the GPU Docker build, place them at:
```
weight/SAM/sam_vit_l_0b3195.pth   # Google Drive ID: 16QARfz1cpumYtwBSf23nlBWtr3hweTQy
weight/YOLO/best.pt                # Google Drive ID: 1maEVUeXS3wCywabZNeO9R-TsBDNanzAS
```

Download with:
```bash
gdown "https://drive.google.com/uc?id=16QARfz1cpumYtwBSf23nlBWtr3hweTQy" -O weight/SAM/sam_vit_l_0b3195.pth
gdown "https://drive.google.com/uc?id=1maEVUeXS3wCywabZNeO9R-TsBDNanzAS" -O weight/YOLO/best.pt
```

## Sample Images

`Sample test/` contains microscope images for manual testing.
