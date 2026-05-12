# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This App Does

XylemVision is a Django web app for automated quantitative analysis of plant root anatomy from microscope cross-section images. It uses a two-stage ML pipeline (YOLO v8 + SAM) to detect and segment xylem vessels, vascular bundles, and total root area, then exports measurements to Excel and CVAT-format training data.

## Running the App

### Docker (GPU) — primary deployment method
```bash
bash build.sh -d        # GPU build, detached
bash build.sh --cpu -d  # CPU-only build, detached
```

The GPU Dockerfile expects model weights to be pre-downloaded locally; the CPU Dockerfile downloads them automatically via `gdown` during build.

App runs at **http://localhost:8000**

### Hot-updating files in the running container

```bash
# Templates only — no reload needed (DEBUG=True re-reads per request)
docker cp analysis/templates/upload.html xylemvision-app:/app/analysis/templates/upload.html

# Python changes — full container restart
docker stop xylemvision-app && docker start xylemvision-app
```

**Avoid `kill -HUP 1` to reload gunicorn:** the new worker boots a second copy of YOLO+SAM onto the GPU before the old worker releases ~6.5 GB. With an 8 GB GPU this OOMs. `docker restart` triggers the same race occasionally — prefer `stop+start` when the container hangs.

On Linux (no Docker Desktop), prefix docker commands with `DOCKER_HOST=unix:///var/run/docker.sock` if the default socket is not found.

### Running locally (development)
```bash
pip install -r requirements.txt
python manage.py runserver
```

No database migrations are needed — the app uses SQLite and has no models.

## Gunicorn configuration (Dockerfile CMD)

```
--workers 1 --threads 4 --timeout 180 --graceful-timeout 30
```

- **`workers=1`** is enforced by GPU memory: each worker loads its own YOLO+SAM (~6.5 GB), and 2 workers won't fit in 8 GB.
- **`threads=4`** lets cheap endpoints (`/status/`, page loads) run concurrently so a slow SAM request can't block status polls. SAM access itself is serialised by `_sam_lock` (`views.py`) because the global `sam_predictor` is not thread-safe.
- **`timeout=180`** — stuck requests die in 3 minutes instead of the previous 1 hour. Without this, one stuck SAM call could appear to "freeze the whole app."

## Architecture

### Request flow
```
Browser upload → /analyze_stream/ (SSE) → engine.progressive_yolo_sam()
                                              ↓
                                      1. YOLO detection
                                      2. Hierarchy post-processing (4 rules)
                                      3. SAM segmentation (hierarchical)
                                      4. Within-class + cross-class overlap resolution
                                      5. Metric calculation
                                              ↓
                                     SSE events → frontend canvas (Fabric.js)
```

### Key files
| File | Role |
|------|------|
| `analysis/engine.py` | ML pipeline: YOLO inference, 4 hierarchy rules, SAM segmentation, metrics |
| `analysis/views.py` | API endpoints; SSE streaming, multi-point SAM prompting, reanalysis, Excel/ZIP export, `_sam_lock` |
| `analysis/utils.py` | Mask blending, contour extraction, SAM prompt generation, **`_resolve_within_class`** + `refine_masks` |
| `analysis/configs.py` | Model paths, device selection, hyperparameters (`YOLO_CONF=0.55`, `ALPHA=0.65`, etc.) |
| `analysis/templates/upload.html` | Single-page UI: Fabric.js canvas, XHR upload + SSE handler, annotation tools, refinement workflow, multi-user status banner |
| `root/settings.py` | Django settings: `DEBUG=True`, `ALLOWED_HOSTS=['*']`, single `analysis` app |

### ML pipeline (`engine.py`)
- **4 Hierarchy Rules**: Filter anatomically impossible YOLO detections (e.g. VB ≥ root, xylem ≥ VB). Rule 4 recovers a missing root detection using Otsu thresholding on the VB-masked image.
- **Hierarchical SAM**: Runs SAM on xylems first → blacks them out → runs on VBs → blacks both out → runs on root. Prevents mask overlap across classes.
- **Within-class overlap resolution** (`utils.py:_resolve_within_class`): After SAM produces all xylem masks, sort by area descending and greedily subtract already-claimed pixels — bigger/more-confident xylems keep contested cell-wall pixels. Same logic for VBs. Without this, adjacent SAM masks share boundary pixels and polygons visually overlap.
- **Coordinate scaling**: Images >2048px are resized for inference. The resize factor is stored in the cache and used to map all SAM coordinates between original-image space (frontend) and resized space (model) in both directions.
- **NMS CPU workaround**: `torchvision.ops.nms` is monkey-patched in `engine.py` to force CPU execution, preventing a CUDA conflict between YOLO and SAM sharing the same GPU.

### Interactive SAM (`/sam_prompt/`)

Interactive segmentation diverges significantly from the batch pipeline. Several mechanisms work together to make a single click on a xylem return only that vessel:

- **Class-aware mask sizing**: The chosen class (`Xylem`, `Vascular bundle`, `Total root`) determines a max-area cap. The cap is derived from the YOLO-detected boxes of that class in the **same image** (`max(yolo_areas) * 2.0`), so the threshold automatically adapts to image resolution and anatomy. Falls back to fractional caps when YOLO didn't detect any.
- **Smallest-mask-containing-click**: With `multimask_output=True`, SAM returns 3 masks at different scales. The default `argmax(scores)` typically returns the largest (whole-root). Instead, pick the smallest mask that (a) contains the click pixel and (b) fits the size cap.
- **Hybrid box+point prompt for xylems**: When clicking with class=Xylem and YOLO detected reference boxes, the click is paired with an auto-generated tight box prompt sized to the median xylem (1.3× median width/height). Box prompts are far stronger than bare points for distinguishing tightly-packed cells.
- **Exclusion mask from existing polygons**: The frontend sends `existing` (all visible same-class polygons in original-image space) with each request. The backend rasterises them via `cv2.fillPoly` in resized space and subtracts from the SAM result, so a new mask cannot bleed into already-segmented neighbours.
- **Multi-point refinement** (`points: [[x, y, label], ...]`): The frontend's "+" / "−" toolbar buttons accumulate positive (label=1) and negative (label=0) point prompts in `activePoints`. Each new click sends the full point list and **replaces** the active polygon with the regenerated mask. Backward-compat: a single legacy `point: [x, y]` is still accepted.
- **Contour-containing-click selection**: SAM masks can be multi-component. After picking the mask, the contour returned to the frontend is the one containing the click pixel (via `cv2.pointPolygonTest`), not necessarily the largest connected component.
- **Adaptive contour simplification**: `cv2.approxPolyDP(epsilon=max(0.5, peri*0.001))` — large polygons get ~0.1% perimeter simplification, small polygons stay detailed.

### Frontend canvas (`upload.html`)

- **Fabric.js viewport**: zoom and pan are clamped via `clampViewport()` so the image never slides off the canvas. Without clamping, panning while zoomed leaves blank space and causes click coordinates to map outside image bounds.
- **Pan vs SAM ordering**: the pan `mouse:up` and the SAM `mouse:up` are merged into a single handler that captures `wasPanning` *before* clearing it. The previous separate handlers fired in registration order, so the SAM handler always saw `isPanning=false` and would fire after every Alt+drag pan in click mode.
- **Polygon styling**: `strokeWidth: 0.5, strokeUniform: true` keeps polygon outlines a constant 0.5 screen-pixels regardless of zoom level. Without `strokeUniform`, strokes scale with zoom and adjacent polygons visually merge.
- **Three click families**: `click` (initial), `click_pos` (refine +), `click_neg` (refine −). Switching to `select` or `box` finalises the active polygon and clears `activePoints`/`activeAnn`. Within the click family, `activePoints` accumulates across clicks.
- **No-cache meta tags** in `<head>` prevent stale-template browser caches from serving outdated JS — added after a debugging session where browsers held onto an old build for hours.

### Cache (`_last_analysis_cache`)

In-memory dict in `views.py` keyed by filename, populated during `analyze_stream_view`. **Lost on container restart.** Used by `/sam_prompt/`, `/reanalyze/`, `/merge_masks/`, `/download_xlsx/`, `/export_training/`, etc.

**Cache-key gotcha**: the SSE *payload* sent to the browser carries `original_image` (a base64 data URL via `pil_to_base64`). The *cache* dict stores the same bytes under `orig_bytes` (raw JPEG bytes). Export endpoints read from the cache, so they must use `orig_bytes`. Confusing the two silently produces empty ZIPs (the `,` check on a base64-style key fails and every image gets skipped).

### Multi-user status visibility
- `_broadcast_status` in `views.py` is a global dict (protected by `_broadcast_lock`) updated during `analyze_stream_view`.
- `GET /status/` returns the current state as JSON.
- Browsers poll `/status/` every 2 seconds; the banner is suppressed on the tab actively uploading (via `_isLocallyUploading` flag in JS).

### API endpoints (`analysis/urls.py`)
- `GET /` — main UI
- `POST /analyze_stream/` — SSE batch analysis
- `GET /status/` — current processing state for multi-user visibility
- `POST /sam_prompt/` — interactive SAM (single point, multi-point, or box; supports `existing` exclusion)
- `POST /reanalyze/` — recalculate metrics from edited polygons
- `POST /merge_masks/` — merge polygon selections
- `POST /download_xlsx/` — single-image Excel export
- `POST /download_all_xlsx/` — batch Excel export (includes Total Root Area column)
- `POST /export_training/` — CVAT-XML + image ZIP (single, reads `cached['orig_bytes']`)
- `POST /export_training_batch/` — CVAT-XML + images ZIP (batch)
- `POST /download_overlays/` — segmentation overlay images ZIP

### Upload & streaming
- The browser uses `XMLHttpRequest` (not `fetch`) for `/analyze_stream/` so `xhr.upload.onprogress` shows real-time upload progress before analysis begins.
- Django buffers all uploaded files to disk before the view runs — the "upload" phase reflects real transfer time, not GPU wait time.
- `DATA_UPLOAD_MAX_NUMBER_FILES = 1000` in `settings.py` (Django default is 100 — batches over 100 images would silently fail without this).

### XLSX export columns
Both single and batch Excel exports produce two sheets:
- **Summary**: Image, Xylem Count, Vascular Bundle Area, Vascular Bundle Diameter, Total Root Area, Total Root Diameter
- **Xylem Details**: Image, Xylem ID, Xylem Area, Xylem Diameter

## Model weights

Not in the repo. For the GPU Docker build, place them at:
```
weight/SAM/sam_vit_l_0b3195.pth   # Google Drive ID: 16QARfz1cpumYtwBSf23nlBWtr3hweTQy
weight/YOLO/best.pt                # Google Drive folder: https://drive.google.com/drive/folders/1ms0JqIBf-lwWWKQiei2fMOhHEIkCxuB2
```

```bash
gdown "https://drive.google.com/uc?id=16QARfz1cpumYtwBSf23nlBWtr3hweTQy" -O weight/SAM/sam_vit_l_0b3195.pth
gdown --folder 1ms0JqIBf-lwWWKQiei2fMOhHEIkCxuB2 -O /tmp/yolo_dl && find /tmp/yolo_dl -name "*.pt" | head -1 | xargs -I{} mv {} weight/YOLO/best.pt && rm -rf /tmp/yolo_dl
```

## Sample images

`Sample test/` contains microscope images for manual testing.
