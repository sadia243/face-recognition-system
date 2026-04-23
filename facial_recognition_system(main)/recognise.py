"""
recognise.py  —  Celebrity face recognition using a trained PCA + SVM model.

Public API
----------
preprocess_image(image_path)          -> np.ndarray  (flat feature vector)
identify_person(image_path, top_n=3)  -> list[dict]
display_result(image_path, results)   -> None  (saves results/recognition_result.png)

CLI:
    python recognise.py path/to/photo.jpg
"""

import sys
import os
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.makedirs('results', exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
MODEL_PATH      = 'model/trained_model.pkl'
CONF_THRESHOLD  = 0.20   # below this the top prediction is flagged as low confidence

# ── Lazy singletons ───────────────────────────────────────────
# Both objects are expensive to create; load once and reuse.
_model        = None
_face_cascade = None


def _get_cascade():
    """Load the Haar cascade classifier once and cache it."""
    global _face_cascade
    if _face_cascade is None:
        xml_path      = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(xml_path)
    return _face_cascade


def _load_model():
    """Load the trained model pickle once and cache it."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. "
                "Please run train.py first."
            )
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        print(f"Model loaded — {_model['n_classes']} people recognised")
    return _model


# ──────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────
def preprocess_image(image_path):
    """
    Full preprocessing pipeline for a single image:

    1. Load with OpenCV (handles JPEG, PNG, BMP, …)
    2. Convert to grayscale
    3. Detect the largest face with the Haar cascade
         - If found  : crop to face + 10 % margin on each side
         - If not found : use the full image as-is
    4. Histogram equalisation — normalises uneven lighting
    5. Resize to the dimensions the model was trained on
    6. Normalise pixels to [0, 1] and flatten to a 1-D vector

    Returns
    -------
    np.ndarray of shape (H*W,), dtype float32
    """
    m    = _load_model()
    h, w = m['image_height'], m['image_width']

    # ── 1 & 2  Load and convert to grayscale ──────────────────
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"OpenCV could not read the image at: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── 3  Haar cascade face detection ────────────────────────
    faces = _get_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,   # how much the image is scaled at each pass
        minNeighbors=5,    # how many neighbours each candidate must have
        minSize=(30, 30),  # smallest detectable face in pixels
    )

    if len(faces) > 0:
        # More than one face found — pick the largest by area
        x, y_pos, fw, fh = max(faces, key=lambda r: r[2] * r[3])

        # Add 10 % margin on every side, clamped to image bounds
        mx = int(fw * 0.10)
        my = int(fh * 0.10)
        x1 = max(0, x - mx)
        y1 = max(0, y_pos - my)
        x2 = min(gray.shape[1], x + fw + mx)
        y2 = min(gray.shape[0], y_pos + fh + my)
        crop = gray[y1:y2, x1:x2]
        print(f"  preprocess: face detected ({fw}x{fh} px) — cropped with margin")
    else:
        # No face found — fall back to the entire image
        crop = gray
        print("  preprocess: no face detected — using full image")

    # ── 4  Histogram equalisation ─────────────────────────────
    crop = cv2.equalizeHist(crop)

    # ── 5  Resize and normalise ───────────────────────────────
    resized  = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    img_norm = resized.astype(np.float32) / 255.0

    # ── 6  HOG feature extraction — parameters must match train.py ─
    features = hog(
        img_norm,
        orientations=8,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
    )
    return features.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# IDENTIFICATION
# ──────────────────────────────────────────────────────────────
def identify_person(image_path, top_n=3):
    """
    Identify the person in the image and return the top-N candidates.

    The best match is *always* returned even when confidence is low.
    When the top confidence is below CONF_THRESHOLD (20 %) the result
    dict includes  low_conf=True  so callers can flag it to the user.

    Parameters
    ----------
    image_path : str
    top_n      : int  (default 3)

    Returns
    -------
    list of dicts, each containing:
        name       : str    — celebrity name (from training labels)
        confidence : float  — probability in [0, 1]
        class_idx  : int    — class index in the model
        low_conf   : bool   — True when rank-1 confidence < CONF_THRESHOLD
    """
    m     = _load_model()
    pca   = m['pca']
    svm   = m['svm']
    names = m['target_names']

    # Preprocess → project into PCA space → get class probabilities
    features     = preprocess_image(image_path)
    features_pca = pca.transform(features.reshape(1, -1))
    probs        = svm.predict_proba(features_pca)[0]

    # Sort classes by probability descending, take top-N
    top_idxs    = np.argsort(probs)[::-1][:top_n]
    best_conf   = float(probs[top_idxs[0]])
    is_low_conf = best_conf < CONF_THRESHOLD

    results = []
    for rank, idx in enumerate(top_idxs):
        results.append({
            'name':       str(names[idx]),
            'confidence': float(probs[idx]),
            'class_idx':  int(idx),
            # Only flag low confidence on the top prediction
            'low_conf':   is_low_conf and rank == 0,
        })

    return results


# ──────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────
def display_result(image_path, results):
    """
    Produce a side-by-side visualisation:
        Col 0  : uploaded photo
        Col 1-3: top-3 matches with sample face + confidence bar

    Saves to results/recognition_result.png and shows interactively.
    """
    m            = _load_model()
    h, w         = m['image_height'], m['image_width']
    sample_faces = m.get('sample_faces', {})

    # Colour scheme
    BG          = '#1a1a2e'
    ACCENT      = ['#00d4ff', '#7bc8f6', '#b8dff8']
    RANK_LABELS = ['1st Match', '2nd Match', '3rd Match']

    fig = plt.figure(figsize=(22, 6), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            height_ratios=[5, 1],
                            hspace=0.10, wspace=0.30)

    # ── Column 0: uploaded photo ───────────────────────────────
    ax_photo = fig.add_subplot(gs[0, 0])
    query    = Image.open(image_path).convert('L')
    ax_photo.imshow(query, cmap='gray', aspect='auto')
    ax_photo.set_title('Uploaded Photo', color='white',
                        fontsize=13, fontweight='bold', pad=8)
    ax_photo.axis('off')
    for spine in ax_photo.spines.values():
        spine.set_edgecolor('#00d4ff')
        spine.set_linewidth(2)

    # Empty bar placeholder under the photo column
    ax_bar0 = fig.add_subplot(gs[1, 0])
    ax_bar0.set_facecolor(BG)
    ax_bar0.axis('off')

    # ── Columns 1-3: top matches ───────────────────────────────
    for i, result in enumerate(results):
        name     = result['name']
        conf     = result['confidence']
        cls_idx  = result['class_idx']
        low_conf = result.get('low_conf', False)
        col      = ACCENT[i]
        bar_col  = '#2ecc71' if conf >= 0.35 else '#f39c12' if conf >= 0.20 else '#e74c3c'

        # Face image panel
        ax_face = fig.add_subplot(gs[0, i + 1])
        ax_face.set_facecolor(BG)

        if cls_idx in sample_faces:
            ax_face.imshow(sample_faces[cls_idx], cmap='gray',
                           aspect='auto', vmin=0, vmax=1)
        else:
            # No sample stored — show a grey placeholder
            ax_face.imshow(np.zeros((h, w)), cmap='gray',
                           aspect='auto', vmin=0, vmax=1)
            ax_face.text(0.5, 0.5, '?', color='white', fontsize=44,
                         ha='center', va='center', transform=ax_face.transAxes)

        # Title: rank + name + optional low-confidence note
        title = f"{RANK_LABELS[i]}\n{name}"
        if low_conf:
            title += "\n(Low confidence)"

        ax_face.set_title(title, color=col, fontsize=10,
                           fontweight='bold', pad=8)
        ax_face.axis('off')

        # Confidence bar panel (row 1)
        ax_bar = fig.add_subplot(gs[1, i + 1])
        ax_bar.set_facecolor(BG)
        ax_bar.barh([0], [1.0],  color='#2a2a4a', height=0.6, zorder=0)  # track
        ax_bar.barh([0], [conf], color=bar_col,   height=0.6, zorder=1)  # fill
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(-0.5, 0.5)
        ax_bar.text(min(conf + 0.03, 0.97), 0, f'{conf:.1%}',
                    color='white', fontsize=10, va='center', fontweight='bold')
        ax_bar.axis('off')

    fig.suptitle('Celebrity Face Recognition Results',
                 color='white', fontsize=15, fontweight='bold', y=1.01)

    out_path = 'results/recognition_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print(f"Result saved to {out_path}")


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recognise.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"\nProcessing: {img_path}")

    matches = identify_person(img_path)

    print("\nTop matches:")
    for i, r in enumerate(matches):
        note = "  [LOW CONFIDENCE]" if r['low_conf'] else ""
        print(f"  #{i + 1}  {r['name']}  ({r['confidence']:.1%}){note}")

    display_result(img_path, matches)
