from .configs import _DILATE_KERN
from pathlib import Path
from .configs import *
from PIL import Image, ImageDraw
import numpy as np
import cv2

def load_image(img_path):
    p = Path(img_path)
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {p}")
    return p, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def group_boxes_by_class(boxes, labels):
    cls2bx = {"Xylem": [], "Vascular bundle": [], "Total root": []}
    for box, label in zip(boxes, labels):
        if label in cls2bx:
            cls2bx[label].append(box)
    return cls2bx

def class_color_cycle(cls: str, k: int) -> tuple[int,int,int]:
    """Color for each mask instance: seeded random for Xylem, fixed cycling for others."""
    if cls == "Xylem":
        rng = np.random.default_rng(k)
        return tuple(int(v) for v in rng.integers(60, 230, size=3))
    base = {
        "Vascular bundle": np.array([0, 255,   0]),
        "Total root":       np.array([230, 120,   0]),
    }[cls]
    factor = 0.85 if k % 2 == 0 else 1.15
    return tuple(int(v) for v in np.clip(base * factor, 0, 255))

def draw_boxes(img: np.ndarray, boxes, labels) -> np.ndarray:
    """Draw bounding boxes and labels onto an RGB image."""
    pil = Image.fromarray(img)
    if len(boxes) > 0:
        dr = ImageDraw.Draw(pil)
        for bx, lb in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, bx)
            dr.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=2)
            dr.text((x1, y1), str(lb), fill=(255, 255, 0))
    return np.array(pil)

def blend_mask(img: np.ndarray, mask: np.ndarray, rgb, alpha=ALPHA) -> np.ndarray:
    """Blend a boolean mask into an RGB image with an outline."""
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c][mask] = (1 - alpha) * out[..., c][mask] + alpha * rgb[c]
    edge = cv2.dilate(mask.astype(np.uint8), _DILATE_KERN, 1).astype(bool) ^ mask
    out[edge] = rgb
    return out.astype(np.uint8)

def prompt_points(box, H, W, n_pos=POS_PTS, neg_edge=NEG_EDGE):
    """Generate SAM point prompts inside and just outside a box."""
    x1, y1, x2, y2 = map(int, box)
    xs = np.linspace(x1 + 5, x2 - 5, int(np.sqrt(n_pos)))
    ys = np.linspace(y1 + 5, y2 - 5, int(np.sqrt(n_pos)))
    pos = np.array([(x, y) for y in ys for x in xs])
    neg = np.array([
        (max(0, x1 - neg_edge), y1),
        (min(W - 1, x2 + neg_edge), y1),
        (x1, max(0, y1 - neg_edge)),
        (x1, min(H - 1, y2 + neg_edge))
    ])
    coords = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return coords, labels

def refine_masks(x_masks, vb_masks, root_masks):
    """Subtract overlaps: remove xylem from VB, and both from root."""
    all_masks = x_masks + vb_masks + root_masks
    if not all_masks:
        return x_masks, vb_masks, root_masks
    shape = all_masks[0].shape

    x_comb = np.zeros(shape, bool)
    for m in x_masks:
        x_comb |= m

    vb_ref  = [m & ~x_comb for m in vb_masks]
    vb_comb = np.zeros(shape, bool)
    for m in vb_ref:
        vb_comb |= m

    root_ref = [m & ~(x_comb | vb_comb) for m in root_masks]
    return x_masks, vb_ref, root_ref

def compute_props(masks, return_contours=False):
    """
    For each boolean mask, compute area and approximate max diameter via
    the minimum enclosing circle. Returns list of dicts:
      {'id': idx, 'area': px_count, 'diameter': px_diameter}.
    If return_contours=True, also includes 'contour': [[x,y], ...].
    """
    props = []
    for idx, m in enumerate(masks):
        cnts, _ = cv2.findContours(
            m.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            entry = {'id': idx, 'area': 0.0, 'diameter': 0.0}
            if return_contours:
                entry['contour'] = []
            props.append(entry)
            continue

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        diameter = 2.0 * radius

        entry = {
            'id': idx,
            'area': float(area),
            'diameter': float(diameter)
        }
        if return_contours:
            entry['contour'] = cnt.reshape(-1, 2).tolist()
        props.append(entry)
    return props


def mask_from_contour(points, H, W):
    """Create a boolean mask from a list of [x, y] polygon points."""
    mask = np.zeros((H, W), dtype=np.uint8)
    if points:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)