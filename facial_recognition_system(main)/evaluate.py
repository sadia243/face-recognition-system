"""
evaluate.py  —  Evaluate the trained PCA + SVM model against the dataset.

Scans 105_classes_pins_dataset/, strips the "pins_" prefix from each folder
name, then applies an 80/20 split (random_state=42) matching train.py exactly.
Preprocessing is identical to train.py:
    grayscale -> histogram equalisation -> resize 64x64 -> normalise -> HOG

Outputs saved to results/:
    confusion_matrix.png
    per_person_accuracy.png
    confidence_distribution.png
    classification_report.txt

Run:
    python evaluate.py
"""

import os
import sys
import pickle
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────
MODEL_PATH  = 'model/trained_model.pkl'
DATASET_DIR = '105_classes_pins_dataset'
VALID_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
CONF_THRESH = 0.20   # low-confidence threshold — matches recognise.py

# HOG parameters — must be identical to train.py and recognise.py
HOG_ORIENTATIONS    = 8
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)

os.makedirs('results', exist_ok=True)

print("=" * 60)
print("CELEBRITY FACE RECOGNITION  —  EVALUATION")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# STEP 1  Load the trained model
# ──────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading model from '{MODEL_PATH}'...")

if not os.path.exists(MODEL_PATH):
    print(f"\n  ERROR: Model not found at '{MODEL_PATH}'.")
    print("  Please run train.py first.")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)

pca          = m['pca']
svm          = m['svm']
target_names = m['target_names']   # e.g. ["Taylor Swift", "Tom Hanks", …]
h            = m['image_height']   # 64
w            = m['image_width']    # 64
n_classes    = m['n_classes']

# Map display name -> class index for folder matching
name_to_idx = {name: idx for idx, name in enumerate(target_names)}

print(f"  Classes : {n_classes}")
print(f"  People  : {', '.join(target_names[:6])}"
      + (" …" if n_classes > 6 else ""))

# ──────────────────────────────────────────────────────────────
# STEP 2  Scan dataset, strip "pins_" prefix, preprocess images
#
#         Pipeline (identical to train.py):
#           BGR -> grayscale -> equalizeHist -> resize(64x64)
#           -> normalise [0,1] -> HOG features
# ──────────────────────────────────────────────────────────────
print(f"\n[2/5] Scanning '{DATASET_DIR}/' and loading images...")

if not os.path.isdir(DATASET_DIR):
    print(f"\n  ERROR: '{DATASET_DIR}/' not found.")
    sys.exit(1)

X_all = []
y_all = []

for raw_folder in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, raw_folder)
    if not os.path.isdir(folder_path):
        continue

    # Strip "pins_" prefix so "pins_Taylor Swift" -> "Taylor Swift"
    display_name = raw_folder.removeprefix('pins_')

    # Skip folders not present in the trained model
    if display_name not in name_to_idx:
        continue

    label_idx = name_to_idx[display_name]
    loaded    = 0

    for img_file in os.listdir(folder_path):
        if os.path.splitext(img_file.lower())[1] not in VALID_EXTS:
            continue

        bgr = cv2.imread(os.path.join(folder_path, img_file))
        if bgr is None:
            continue

        # Preprocessing — 100 % identical to train.py loading loop
        gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eq       = cv2.equalizeHist(gray)
        resized  = cv2.resize(eq, (w, h), interpolation=cv2.INTER_LANCZOS4)
        img_norm = resized.astype(np.float32) / 255.0
        features = hog(img_norm,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK)

        X_all.append(features.astype(np.float32))
        y_all.append(label_idx)
        loaded += 1

    print(f"  {display_name}: {loaded} image(s)")

if not X_all:
    print("\n  ERROR: No images loaded — check that the dataset folder exists "
          "and matches the classes in the trained model.")
    sys.exit(1)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.int32)
print(f"\n  Total loaded: {len(X_all)} images across {len(np.unique(y_all))} people")

# ──────────────────────────────────────────────────────────────
# STEP 3  Reproduce the 80/20 split, apply PCA, predict
# ──────────────────────────────────────────────────────────────
print("\n[3/5] Reproducing 80/20 split (random_state=42) and predicting...")

try:
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
except ValueError:
    # Fallback when a class has too few samples to stratify
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

# Project HOG features into PCA space, then predict
X_test_pca = pca.transform(X_test)
y_pred     = svm.predict(X_test_pca)
y_prob     = svm.predict_proba(X_test_pca)

overall_acc = accuracy_score(y_test, y_pred)
print(f"  Test samples    : {len(X_test)}")
print(f"  Overall accuracy: {overall_acc:.2%}")

# ──────────────────────────────────────────────────────────────
# STEP 4  Generate and save evaluation charts
# ──────────────────────────────────────────────────────────────
print("\n[4/5] Generating evaluation charts...")

# ── 4a  Confusion matrix ──────────────────────────────────────
print("  [4a] Confusion matrix...")

cm      = confusion_matrix(y_test, y_pred)
fs      = max(6, 10 - n_classes // 5)          # dynamic font size
fig_dim = max(8, n_classes * max(0.7, 10 / n_classes))
fig, ax = plt.subplots(figsize=(fig_dim, fig_dim * 0.85))

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ticks = np.arange(n_classes)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=fs)
ax.set_yticklabels(target_names, fontsize=fs)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                fontsize=max(5, fs - 1),
                color='white' if cm[i, j] > thresh else 'black')

ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual',    fontsize=12)
ax.set_title('Confusion Matrix — Test Set', fontsize=15,
             fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: results/confusion_matrix.png")

# ── 4b  Per-person accuracy bar chart ────────────────────────
print("  [4b] Per-person accuracy chart...")

per_person = []
for cls in range(n_classes):
    mask = y_test == cls
    if mask.any():
        acc   = accuracy_score(y_test[mask], y_pred[mask])
        count = int(mask.sum())
        per_person.append((target_names[cls], acc, count))

per_person.sort(key=lambda x: x[1], reverse=True)
pp_names  = [r[0] for r in per_person]
pp_accs   = [r[1] for r in per_person]
pp_counts = [r[2] for r in per_person]

bar_colors = [
    '#2ecc71' if a >= 0.80 else '#f39c12' if a >= 0.60 else '#e74c3c'
    for a in pp_accs
]

fig, ax = plt.subplots(figsize=(14, max(6, len(pp_names) * 0.55)))
bars = ax.barh(pp_names, pp_accs, color=bar_colors,
               edgecolor='white', linewidth=0.4)

ax.axvline(overall_acc, color='navy', linestyle='--', linewidth=1.5,
           label=f'Overall: {overall_acc:.1%}')
ax.set_xlim(0, 1.22)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Per-Person Recognition Accuracy', fontsize=15,
             fontweight='bold', pad=12)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

for bar, acc, cnt in zip(bars, pp_accs, pp_counts):
    ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{acc:.1%}  (n={cnt})', va='center', fontsize=9)

legend_patches = [
    Patch(color='#2ecc71', label='>= 80 %'),
    Patch(color='#f39c12', label='60-79 %'),
    Patch(color='#e74c3c', label='< 60 %'),
]
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=legend_patches + handles,
    labels=[p.get_label() for p in legend_patches] + labels,
    loc='lower right', fontsize=9,
)
plt.tight_layout()
plt.savefig('results/per_person_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: results/per_person_accuracy.png")

# ── 4c  Confidence distribution histogram ────────────────────
print("  [4c] Confidence distribution chart...")

best_probs = y_prob[np.arange(len(y_test)), y_pred]
correct    = y_pred == y_test

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, 1, 26)
ax.hist(best_probs[correct],  bins=bins, alpha=0.7,
        color='#2ecc71', label='Correct predictions')
ax.hist(best_probs[~correct], bins=bins, alpha=0.7,
        color='#e74c3c', label='Wrong predictions')
ax.axvline(CONF_THRESH, color='navy', linestyle='--', linewidth=1.5,
           label=f'Low-confidence threshold ({CONF_THRESH:.0%})')
ax.set_xlabel('Prediction Confidence', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Confidence Distribution: Correct vs Wrong Predictions',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: results/confidence_distribution.png")

# ──────────────────────────────────────────────────────────────
# STEP 5  Classification report
# ──────────────────────────────────────────────────────────────
print("\n[5/5] Classification report...")

report = classification_report(y_test, y_pred, target_names=target_names)
print("\n" + report)

report_path = 'results/classification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("CELEBRITY FACE RECOGNITION  —  CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Overall accuracy : {overall_acc:.2%}\n")
    f.write(f"Test samples     : {len(y_test)}\n")
    f.write(f"Classes          : {n_classes}\n\n")
    f.write(report)

print(f"  Saved: {report_path}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print(f"  Overall accuracy : {overall_acc:.2%}")
print("  Saved to results/:")
print("    confusion_matrix.png")
print("    per_person_accuracy.png")
print("    confidence_distribution.png")
print("    classification_report.txt")
print("=" * 60)
