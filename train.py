"""
train.py  —  Train a PCA + SVM celebrity face recognition model.

Expects the 105_classes_pins_dataset/ folder where every subfolder is one person.
Subfolder names may carry a "pins_" prefix (e.g. pins_Taylor Swift); that prefix
is stripped automatically so labels read "Taylor Swift" in all outputs.

Run:
    python train.py
"""

import os
import sys
import pickle
import numpy as np
import cv2
import scipy.ndimage
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.feature import hog

# ── Configuration ──────────────────────────────────────────────
DATASET_DIR = '105_classes_pins_dataset'
MODEL_PATH  = 'model/trained_model.pkl'
IMG_H       = 64                                # training image height (pixels)
IMG_W       = 64                                # training image width  (pixels)
MIN_IMAGES        = 2                           # skip people with fewer images
TOP_N_CELEBRITIES = 20                          # keep only the N busiest folders
VALID_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

os.makedirs('model',   exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("CELEBRITY FACE RECOGNITION  —  TRAINING")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# STEP 1  Validate the dataset folder
# ──────────────────────────────────────────────────────────────
print(f"\n[1/8] Checking dataset folder: '{DATASET_DIR}/'")

if not os.path.isdir(DATASET_DIR):
    print(f"\n  ERROR: '{DATASET_DIR}/' folder not found.")
    print("  Please add your dataset to the dataset/ folder first.")
    print("  Each subfolder should be named after the person it contains.\n")
    print("  Example structure:")
    print("    dataset/")
    print("    ├── Tom_Hanks/")
    print("    │   ├── photo1.jpg")
    print("    │   └── photo2.jpg")
    print("    └── Elon_Musk/")
    print("        └── photo1.jpg\n")
    sys.exit(1)

# Collect all subfolders (one per celebrity)
all_subfolders = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

if not all_subfolders:
    print(f"\n  ERROR: '{DATASET_DIR}/' folder is empty.")
    print("  Please add your dataset to the dataset/ folder first.")
    print("  Each subfolder should be named after the person it contains.\n")
    sys.exit(1)

print(f"  Found {len(all_subfolders)} subfolder(s) — selecting top {TOP_N_CELEBRITIES} by image count")

# Count valid images in every subfolder, then keep the top N
folder_counts = []
for folder_name in all_subfolders:
    folder_path = os.path.join(DATASET_DIR, folder_name)
    count = sum(
        1 for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in VALID_EXTS
    )
    folder_counts.append((folder_name, count))

# Sort descending by image count and slice
folder_counts.sort(key=lambda x: x[1], reverse=True)
selected = folder_counts[:TOP_N_CELEBRITIES]

print(f"\n  Top {TOP_N_CELEBRITIES} celebrities selected:")
for rank, (name, cnt) in enumerate(selected, 1):
    display = name.removeprefix('pins_')
    print(f"    {rank:>2}. {display:<30} {cnt} images")

# Replace all_subfolders with only the selected names (order preserved)
all_subfolders = [name for name, _ in selected]

# ──────────────────────────────────────────────────────────────
# STEP 2  Load images from each person's folder
#         Preprocessing: grayscale → histogram equalisation → resize
# ──────────────────────────────────────────────────────────────
print(f"\n[2/8] Loading and preprocessing images "
      f"(grayscale + hist-eq, {IMG_H}×{IMG_W} px)...")

X            = []   # flat feature vectors
y            = []   # integer class labels
target_names = []   # class index → person name
skipped      = []   # people with too few images

for folder_name in all_subfolders:
    folder_path = os.path.join(DATASET_DIR, folder_name)

    # Gather image files in this subfolder
    img_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in VALID_EXTS
    ]

    if len(img_files) < MIN_IMAGES:
        skipped.append(f"{folder_name}  ({len(img_files)} image(s), need >= {MIN_IMAGES})")
        continue

    label_idx = len(target_names)   # next available class index
    # Strip the "pins_" prefix so "pins_Taylor Swift" → "Taylor Swift"
    display_name = folder_name.removeprefix('pins_')
    target_names.append(display_name)
    loaded = 0

    for img_file in img_files:
        img_path = os.path.join(folder_path, img_file)
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue  # skip corrupt or unreadable file

        # Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Histogram equalisation — normalises lighting across dataset
        eq = cv2.equalizeHist(gray)

        # Resize to fixed training dimensions
        resized = cv2.resize(eq, (IMG_W, IMG_H), interpolation=cv2.INTER_LANCZOS4)

        # Flatten to 1-D and normalise pixel values to [0, 1]
        X.append(resized.flatten().astype(np.float32) / 255.0)
        y.append(label_idx)
        loaded += 1

    print(f"  {folder_name}: {loaded} image(s) loaded")

# Report skipped people
if skipped:
    print("\n  Skipped (too few images):")
    for s in skipped:
        print(f"    - {s}")

# Guard: nothing loaded
if not X:
    print("\n  ERROR: No images could be loaded from the dataset.")
    print("  Please add your dataset to the dataset/ folder first.")
    sys.exit(1)

# Guard: need at least 2 people for classification
n_classes = len(target_names)
if n_classes < 2:
    print(f"\n  ERROR: Need at least 2 people in the dataset, found {n_classes}.")
    sys.exit(1)

# Convert to NumPy arrays
X            = np.array(X, dtype=np.float32)
y            = np.array(y, dtype=np.int32)
target_names = np.array(target_names)

print(f"\n  Total loaded : {len(X)} images | {n_classes} people")
avg_per_person = len(X) / n_classes
if avg_per_person < 10:
    print(f"  Tip: aim for >= 10 images per person for better accuracy "
          f"(current average: {avg_per_person:.1f})")

# ──────────────────────────────────────────────────────────────
# STEP 3  Train / test split  (80 % / 20 %, stratified)
# ──────────────────────────────────────────────────────────────
print("\n[3/8] Splitting data (80 % train / 20 % test, random_state=42)...")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # Fallback: some class may have only 1 sample — disable stratify
    print("  Warning: could not stratify split (some class too small); "
          "using random split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# Save one representative (un-augmented) face per class for the GUI display
sample_faces = {}
for cls in range(n_classes):
    mask = y_train == cls
    if mask.any():
        sample_faces[cls] = X_train[mask][0].reshape(IMG_H, IMG_W)

# ──────────────────────────────────────────────────────────────
# STEP 4  Data augmentation  —  triple the training set
#         Original + geometric copy + photometric copy
# ──────────────────────────────────────────────────────────────
print("\n[4/8] Augmenting training data to 3x size...")
print("  Geometric   : rotation ±15 deg, random horizontal flip")
print("  Photometric : brightness x[0.7-1.3], Gaussian noise (sigma=0.02)")

rng = np.random.default_rng(seed=0)   # fixed seed for reproducibility

def _augment(X_flat, h, w, mode):
    """Return one augmented copy of a flat image array."""
    imgs = X_flat.reshape(-1, h, w)
    out  = []
    for img in imgs:
        if mode == 'geom':
            angle = rng.uniform(-15, 15)
            aug   = scipy.ndimage.rotate(img, angle, reshape=False, mode='nearest')
            if rng.random() > 0.5:
                aug = np.fliplr(aug)
        else:
            # Photometric: brightness multiplier + additive Gaussian noise
            factor = rng.uniform(0.7, 1.3)
            aug    = np.clip(img * factor, 0.0, 1.0)
            noise  = rng.normal(0.0, 0.02, aug.shape)
            aug    = np.clip(aug + noise, 0.0, 1.0)
        out.append(aug.flatten())
    return np.array(out, dtype=np.float32)

aug_geom  = _augment(X_train, IMG_H, IMG_W, 'geom')
aug_photo = _augment(X_train, IMG_H, IMG_W, 'photo')

X_train_aug = np.vstack([X_train, aug_geom, aug_photo])
y_train_aug = np.concatenate([y_train, y_train, y_train])

print(f"  {len(X_train)} -> {len(X_train_aug)} training samples")

# ──────────────────────────────────────────────────────────────
# STEP 5  HOG feature extraction
#         Augmentation runs in pixel space first; HOG is then
#         computed from every image (original + both augmented
#         copies) to produce lighting-robust gradient descriptors.
#         These HOG vectors replace raw pixels as PCA input.
# ──────────────────────────────────────────────────────────────
print("\n[5/8] Extracting HOG features from all training and test images...")

# HOG parameters — must be identical across train.py, recognise.py, evaluate.py
HOG_ORIENTATIONS    = 8      # number of gradient orientation bins
HOG_PIXELS_PER_CELL = (4, 4) # pixel area covered by each histogram cell
HOG_CELLS_PER_BLOCK = (2, 2) # cells grouped per normalisation block


def _extract_hog(flat_images, h, w):
    """Return a HOG feature matrix: one row per flat normalised image."""
    imgs = flat_images.reshape(-1, h, w)
    return np.array(
        [hog(img,
             orientations=HOG_ORIENTATIONS,
             pixels_per_cell=HOG_PIXELS_PER_CELL,
             cells_per_block=HOG_CELLS_PER_BLOCK)
         for img in imgs],
        dtype=np.float32,
    )


X_train_hog = _extract_hog(X_train_aug, IMG_H, IMG_W)
X_test_hog  = _extract_hog(X_test,      IMG_H, IMG_W)

print(f"  HOG feature length  : {X_train_hog.shape[1]:,}")
print(f"  Training HOG matrix : {X_train_hog.shape}")

# ──────────────────────────────────────────────────────────────
# STEP 6  PCA — compress HOG descriptors into compact components
#         Clamp n_components so it never exceeds data dimensions
# ──────────────────────────────────────────────────────────────
print("\n[6/8] Fitting PCA on HOG features (up to 150 components, whiten=True)...")

n_components = min(150, X_train_hog.shape[0] - 1, X_train_hog.shape[1])
pca          = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca  = pca.fit_transform(X_train_hog)
X_test_pca   = pca.transform(X_test_hog)

print(f"  Components used     : {n_components}")
print(f"  Variance explained  : {pca.explained_variance_ratio_.sum():.2%}")

# ──────────────────────────────────────────────────────────────
# STEP 6  SVM classifier with GridSearchCV
#         RBF kernel, class-balanced weights for uneven datasets
# ──────────────────────────────────────────────────────────────
print("\n[7/8] Training SVM with GridSearchCV (3-fold CV)...")
print("  C     : [10, 100, 1000]")
print("  gamma : [0.001, 0.01]")
print("  Using all CPU cores (n_jobs=-1)")
print("  This may take several minutes...\n")

param_grid = {
    'C':     [10, 100, 1000],
    'gamma': [0.001, 0.01],
}

base_svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
grid     = GridSearchCV(base_svm, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train_pca, y_train_aug)

print(f"\n  Best params     : {grid.best_params_}")
print(f"  Best CV score   : {grid.best_score_:.2%}")

test_acc = grid.score(X_test_pca, y_test)
print(f"  Test accuracy   : {test_acc:.2%}")

# ──────────────────────────────────────────────────────────────
# STEP 7  Save everything needed for recognition and evaluation
# ──────────────────────────────────────────────────────────────
print(f"\n[8/8] Saving model to '{MODEL_PATH}'...")

model_data = {
    'pca':          pca,               # fitted PCA transformer
    'svm':          grid.best_estimator_,  # best SVM found by grid search
    'target_names': target_names,      # array of person names (index = class)
    'image_height': IMG_H,             # expected input image height
    'image_width':  IMG_W,             # expected input image width
    'n_classes':    n_classes,         # total number of people
    'X_test':       X_test_hog,        # held-out test HOG features
    'y_test':       y_test,            # held-out test labels
    'sample_faces': sample_faces,      # representative face per class for display
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_data, f)

print("  Saved successfully.")
print("\n" + "=" * 60)
print(f"TRAINING COMPLETE")
print(f"  People       : {n_classes}")
print(f"  Test accuracy: {test_acc:.2%}")
print(f"  Model saved  : {MODEL_PATH}")
print("=" * 60)
