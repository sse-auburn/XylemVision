#!/usr/bin/env python3
import numpy as np
import torch
import torchvision.ops as tv_ops
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from analysis.configs import *
from analysis.utils import *

# ── NMS CPU workaround (prevents CUDA NMS conflict between YOLO and SAM) ────
_orig_nms = tv_ops.nms

def _nms_cpu(boxes, scores, iou_threshold):
    keep = _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold)
    return keep.to(boxes.device)

setattr(tv_ops, 'nms', _nms_cpu)

# ── Model handles (loaded lazily on first inference call) ────────────────────
yolo_model    = None
_sam          = None
sam_predictor = None


def _ensure_models():
    global yolo_model, _sam, sam_predictor
    if yolo_model is not None:
        return
    yolo_model    = YOLO(YOLO_WEIGHTS).to(DEVICE).eval()
    _sam          = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE).eval()
    sam_predictor = SamPredictor(_sam)
    print("✅ Models ready on", DEVICE)


def get_sam_predictor():
    _ensure_models()
    return sam_predictor

# ── Constants ────────────────────────────────────────────────────────────────
YOLO_CONF        = 0.55
VB_TO_ROOT_RATIO = 3.0
MAX_IMG_SIDE     = 2048   # resize large images before processing (speed + memory)


# ── Box geometry helpers ─────────────────────────────────────────────────────
def _box_area(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _overlap_fraction(outer, inner):
    ix1 = max(outer[0], inner[0]); iy1 = max(outer[1], inner[1])
    ix2 = min(outer[2], inner[2]); iy2 = min(outer[3], inner[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = _box_area(inner)
    return inter / inner_area if inner_area > 0 else 0.0


# ── Rules 1-3: hierarchy post-processing ────────────────────────────────────
def postprocess_detections(cls2bx):
    """Apply hierarchy rules to YOLO boxes before SAM segmentation."""
    xylems = list(cls2bx["Xylem"])
    vbs    = list(cls2bx["Vascular bundle"])
    roots  = list(cls2bx["Total root"])

    # Rule 3: no root, two VBs — if one is ≥3× the other, reclassify as root
    if not roots and len(vbs) == 2:
        a0, a1 = _box_area(vbs[0]), _box_area(vbs[1])
        ratio  = max(a0, a1) / max(min(a0, a1), 1.0)
        if ratio >= VB_TO_ROOT_RATIO:
            bigger  = 0 if a0 >= a1 else 1
            roots   = [vbs[bigger]]
            vbs     = [vbs[1 - bigger]]

    # Rule 2: two VBs — keep the one that best satisfies hierarchy
    if len(vbs) == 2:
        if roots:
            root_area = _box_area(roots[0])
            scores = []
            for vb in vbs:
                vb_area     = _box_area(vb)
                area_ok     = 1.0 if vb_area < root_area else 0.0
                inside_frac = _overlap_fraction(roots[0], vb)
                xylem_cover = (sum(_overlap_fraction(vb, x) for x in xylems) / len(xylems)
                               if xylems else 0.0)
                scores.append(area_ok * 2 + inside_frac + xylem_cover)
            vbs = [vbs[int(np.argmax(scores))]]
        else:
            a0, a1 = _box_area(vbs[0]), _box_area(vbs[1])
            vbs    = [vbs[0 if a0 <= a1 else 1]]

    # Rule 1: drop VBs not smaller than root
    if roots:
        root_area = _box_area(roots[0])
        vbs = [vb for vb in vbs if _box_area(vb) < root_area]

    # Rule 1: drop xylems not smaller than the smallest VB or root
    ref_areas = [_box_area(b) for b in vbs + roots]
    if ref_areas:
        min_ref = min(ref_areas)
        xylems  = [x for x in xylems if _box_area(x) < min_ref]

    return {"Xylem": xylems, "Vascular bundle": vbs, "Total root": roots}


# ── Rule 4: recover root when missing but VB is present ─────────────────────
def rule4_find_root(gray_img, cls2bx, rgb_img):
    """
    When no Total root is detected but VBs exist:
    1. SAM on each VB box → combined VB mask
    2. Fill VB pixels with white (255)
    3. Otsu threshold → dark pixels = root tissue
    4. CCA pass 1: mean-area noise filter
    5. 7×7 ellipse dilation to close gaps
    6. CCA pass 2: largest component → root bounding box
    """
    vbs = cls2bx["Vascular bundle"]
    if not vbs:
        return None

    H, W = gray_img.shape[:2]
    sam_predictor.set_image(rgb_img)
    vb_mask = np.zeros((H, W), dtype=bool)
    for bx in vbs:
        m = _choose_mask(np.array(bx, dtype=float))
        vb_mask |= m

    filled = gray_img.copy()
    filled[vb_mask] = 255
    _, binary = cv2.threshold(filled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    n, label_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    fg = stats[1:]
    areas = fg[:, cv2.CC_STAT_AREA].astype(float)
    mean_area = float(np.mean(areas))
    filtered = np.zeros_like(binary)
    for idx in np.where(areas >= mean_area)[0]:
        filtered[label_map == (idx + 1)] = 255

    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(filtered, kern, iterations=1)

    n2, _, stats2, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n2 <= 1:
        return None

    fg2   = stats2[1:]
    best  = int(np.argmax(fg2[:, cv2.CC_STAT_AREA]))
    lx    = int(fg2[best, cv2.CC_STAT_LEFT]);  lw = int(fg2[best, cv2.CC_STAT_WIDTH])
    ly    = int(fg2[best, cv2.CC_STAT_TOP]);   lh = int(fg2[best, cv2.CC_STAT_HEIGHT])
    return np.array([lx, ly, lx + lw, ly + lh], dtype=float)


# ── SAM helpers ──────────────────────────────────────────────────────────────
def _choose_mask(box_np, pts=None, pt_labels=None):
    masks, scores, _ = sam_predictor.predict(
        box=box_np, point_coords=pts, point_labels=pt_labels, multimask_output=True
    )
    return masks[scores.squeeze().argmax()]


def _segment_in_box(box_np, pad=0.12, min_fill=0.05):
    """Box-prompted SAM for one compact structure (xylem / vascular bundle),
    hardened against the two failure modes seen on low-resolution images:

    * **bleed** — SAM's mask leaks into neighbouring vessels / background.
      Every candidate mask is clipped to the YOLO box (+ `pad`), so nothing
      well outside the detected box survives.
    * **shrink / wrong scale** — a bare box prompt sometimes locks onto a
      sub-part. A positive point at the box centre anchors SAM on the boxed
      object, and selection prefers SAM's highest-confidence mask that still
      fills a meaningful fraction of the box (not a hairline sliver).
    """
    bx1, by1 = float(min(box_np[0], box_np[2])), float(min(box_np[1], box_np[3]))
    bx2, by2 = float(max(box_np[0], box_np[2])), float(max(box_np[1], box_np[3]))
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    masks, scores, _ = sam_predictor.predict(
        box=box_np,
        point_coords=np.array([[cx, cy]], dtype=float),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    sc = scores.squeeze()
    H, W = masks.shape[1], masks.shape[2]
    pad_x, pad_y = pad * (bx2 - bx1), pad * (by2 - by1)
    ix1 = max(0, int(bx1 - pad_x)); iy1 = max(0, int(by1 - pad_y))
    ix2 = min(W, int(round(bx2 + pad_x))); iy2 = min(H, int(round(by2 + pad_y)))
    region = np.zeros((H, W), dtype=bool); region[iy1:iy2, ix1:ix2] = True
    box_area = max(1.0, float((iy2 - iy1) * (ix2 - ix1)))

    cand = []
    for i in range(len(masks)):
        m = masks[i] & region                    # clip bleed to the box
        a = int(m.sum())
        if a >= 12:
            cand.append((float(sc[i]), a, m))
    if not cand:
        return masks[int(sc.argmax())]
    substantial = [c for c in cand if c[1] >= min_fill * box_area]
    if substantial:
        return max(substantial, key=lambda c: c[0])[2]   # best SAM score, not a sliver
    return max(cand, key=lambda c: c[1])[2]               # else the largest available


def _segment_xylem_per_crop(box_resized, rgb_orig, scale, H_res, W_res, pad=2.0):
    """SAM on a full-resolution crop centred on one xylem box.

    Gives SAM much more pixel detail per xylem than loading the entire
    downscaled image.  Returns a boolean mask in the RESIZED (2048px) space
    so the rest of the pipeline (no_x blackout, overlay, metrics) is unchanged.
    """
    if scale != 1.0:
        bx1_o = box_resized[0] / scale
        by1_o = box_resized[1] / scale
        bx2_o = box_resized[2] / scale
        by2_o = box_resized[3] / scale
    else:
        bx1_o, by1_o, bx2_o, by2_o = (float(v) for v in box_resized)

    bw = bx2_o - bx1_o
    bh = by2_o - by1_o
    H_orig, W_orig = rgb_orig.shape[:2]

    cx1 = max(0,      int(bx1_o - pad * bw))
    cy1 = max(0,      int(by1_o - pad * bh))
    cx2 = min(W_orig, int(round(bx2_o + pad * bw)))
    cy2 = min(H_orig, int(round(by2_o + pad * bh)))

    crop_rgb = rgb_orig[cy1:cy2, cx1:cx2]

    box_in_crop = np.array([
        bx1_o - cx1, by1_o - cy1,
        bx2_o - cx1, by2_o - cy1,
    ], dtype=float)

    sam_predictor.set_image(crop_rgb)
    mask_in_crop = _segment_in_box(box_in_crop)  # bool, crop-orig-res space
    torch.cuda.empty_cache()  # release per-crop encoder memory to avoid fragmentation

    # Map mask back to the resized (2048px) space
    rx1 = int(round(cx1 * scale)); ry1 = int(round(cy1 * scale))
    rx2 = min(W_res, int(round(cx2 * scale))); ry2 = min(H_res, int(round(cy2 * scale)))
    tw, th = rx2 - rx1, ry2 - ry1
    if tw <= 0 or th <= 0:
        return np.zeros((H_res, W_res), dtype=bool)

    mask_region = cv2.resize(
        mask_in_crop.astype(np.uint8) * 255, (tw, th),
        interpolation=cv2.INTER_NEAREST,
    ) > 127

    full_mask = np.zeros((H_res, W_res), dtype=bool)
    full_mask[ry1:ry2, rx1:rx2] = mask_region
    return full_mask


def robust_root_mask(img_rgb, masked_rgb, box, H, W):
    box_np = np.array(box, float)
    sam_predictor.set_image(masked_rgb)
    m1 = _choose_mask(box_np)
    if m1.sum() >= MIN_ROOT_PX:
        return m1
    pts, lbls = prompt_points(box, H, W)
    sam_predictor.set_image(masked_rgb)
    m2 = _choose_mask(box_np, pts, lbls)
    if m2.sum() >= MIN_ROOT_PX:
        return m2
    sam_predictor.set_image(img_rgb)
    return _choose_mask(box_np, pts, lbls)


# ── Metrics ──────────────────────────────────────────────────────────────────
def calculate_metrics(x_masks, vb_masks, root_masks):
    x_props    = compute_props(x_masks)
    vb_props   = compute_props(vb_masks)
    root_props = compute_props(root_masks)
    return {
        'xylem_count':       len(x_props),
        'vb_total_area':     sum(p['area']     for p in vb_props),
        'vb_max_diameter':   max((p['diameter'] for p in vb_props),   default=0.0),
        'root_total_area':   sum(p['area']     for p in root_props),
        'root_max_diameter': max((p['diameter'] for p in root_props), default=0.0),
        'xylem_details':     x_props,
    }


# ── Resize helper ────────────────────────────────────────────────────────────
def _resize_to_max(bgr, max_side=MAX_IMG_SIDE):
    h, w    = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, 1.0
    scale = max_side / longest
    return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


# ── Main pipeline ────────────────────────────────────────────────────────────
def progressive_yolo_sam(pil_image):
    _ensure_models()
    rgb_orig = np.array(pil_image)
    bgr_orig = rgb_orig[..., ::-1]

    # Resize for faster YOLO + SAM processing
    bgr,  scale = _resize_to_max(bgr_orig)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Stage 1: YOLO detection (conf=0.55)
    with torch.no_grad():
        det    = yolo_model.predict(bgr, verbose=False, conf=YOLO_CONF)[0]
        boxes  = det.boxes.xyxy.cpu().numpy()
        labels = [yolo_model.model.names[int(i)] for i in det.boxes.cls.cpu().numpy()]

    cls2bx = {"Xylem": [], "Vascular bundle": [], "Total root": []}
    for bx, lb in zip(boxes, labels):
        if lb in cls2bx:
            cls2bx[lb].append(bx)

    # Rules 1-3: hierarchy filtering
    cls2bx = postprocess_detections(cls2bx)

    # Rule 4: recover root when missing
    if not cls2bx["Total root"] and cls2bx["Vascular bundle"]:
        r4_box = rule4_find_root(gray, cls2bx, rgb)
        if r4_box is not None:
            cls2bx["Total root"] = [r4_box]

    # Flat boxes/labels after post-processing (for overlay drawing)
    all_boxes  = np.array([bx for cls_name in ["Xylem", "Vascular bundle", "Total root"]
                           for bx in cls2bx[cls_name]], dtype=float) if any(cls2bx.values()) else np.empty((0, 4))
    all_labels = [cls_name for cls_name in ["Xylem", "Vascular bundle", "Total root"]
                  for _ in cls2bx[cls_name]]

    # Stage 2: SAM segmentation (hierarchical masking)
    # Xylems: per-object full-res crop → better detail for small vessels.
    # Hardened against GPU OOM on dense/high-res roots: each crop's encoder pass
    # needs ~1GB transient and many crops can fragment the 8GB GPU. Any crop that
    # OOMs falls back to whole-image box SAM so the worker never crashes; order is
    # preserved (placeholders) so xylem colours/IDs stay stable.
    torch.cuda.empty_cache()
    x_masks  = [None] * len(cls2bx["Xylem"])
    _oom_idx = []
    for k, bx in enumerate(cls2bx["Xylem"]):
        try:
            x_masks[k] = _segment_xylem_per_crop(np.array(bx, float), rgb_orig, scale, H, W)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _oom_idx.append(k)
    if _oom_idx:
        whole_ok = False
        try:
            sam_predictor.set_image(rgb)
            whole_ok = True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        for k in _oom_idx:
            mask = None
            if whole_ok:
                try:
                    mask = _segment_in_box(np.array(cls2bx["Xylem"][k], float))
                except Exception:
                    torch.cuda.empty_cache()
            x_masks[k] = mask if mask is not None else np.zeros((H, W), dtype=bool)

    no_x = rgb.copy()
    for m in x_masks:
        no_x[m] = 0

    sam_predictor.set_image(no_x)
    vb_masks = [_segment_in_box(np.array(bx, float)) for bx in cls2bx["Vascular bundle"]]

    no_vb = no_x.copy()
    for m in vb_masks:
        no_vb[m] = 0

    root_masks = [robust_root_mask(rgb, no_vb, bx, H, W) for bx in cls2bx["Total root"]]

    if x_masks or vb_masks or root_masks:
        x_masks, vb_masks, root_masks = refine_masks(x_masks, vb_masks, root_masks)

    metrics = calculate_metrics(x_masks, vb_masks, root_masks)

    # Scale metrics back to original image pixel dimensions
    if scale != 1.0:
        s2 = scale * scale
        metrics['vb_total_area']     /= s2
        metrics['vb_max_diameter']   /= scale
        metrics['root_total_area']   /= s2
        metrics['root_max_diameter'] /= scale
        for p in metrics['xylem_details']:
            p['area']     /= s2
            p['diameter'] /= scale

    # Extract polygon contours — scale back to original image coordinates
    def _scaled_contours(masks):
        props = compute_props(masks, return_contours=True)
        if scale == 1.0:
            return [p['contour'] for p in props]
        return [[[round(x/scale), round(y/scale)] for x,y in p['contour']] for p in props]

    x_contours  = _scaled_contours(x_masks)
    vb_contours = _scaled_contours(vb_masks)
    rt_contours = _scaled_contours(root_masks)

    # Build colour overlay
    colour_meta = []
    overlay = rgb.copy()
    for cls_name, masks in [('Xylem', x_masks), ('Vascular bundle', vb_masks), ('Total root', root_masks)]:
        for k, m in enumerate(masks):
            col = class_color_cycle(cls_name, k)
            overlay = blend_mask(overlay, m, col)
            colour_meta.append({'class': cls_name, 'inst': k, 'rgb': list(col)})

    if len(all_boxes):
        overlay = draw_boxes(overlay, all_boxes, all_labels)
    overlay_pil = Image.fromarray(overlay)

    return {
        'file':     'in_memory_image',
        'n_xylem':  len(x_masks),
        'n_vb':     len(vb_masks),
        'n_root':   len(root_masks),
        'metrics':  metrics,
        'colours':  colour_meta,
        'boxes':    [[float(v) for v in b] for b in all_boxes],
        'labels':   all_labels,
        'contours': {
            'Xylem':           x_contours,
            'Vascular bundle': vb_contours,
            'Total root':      rt_contours,
        },
        'rgb_np': rgb,       # resized image — used by sam_prompt_view
        'scale':  scale,     # resize factor — used to convert frontend coords
    }, pil_image, overlay_pil
