#  Celebrity Face Recognition System

**Module:** Computer Vision and AI | CMS22202  
**Level:** Level 5 | Ravensbourne University London  
**Dataset:** Pins Face Recognition (Kaggle) — 17,534 images, 20 celebrities

---

## What This Does

This system takes any photo and identifies which celebrity it most closely matches. It uses a machine learning pipeline built entirely in Python — from raw image data through to a live graphical interface where you can upload photos and see results in real time.

---

## How It Works

```
Photo → Grayscale + Resize → HOG Features → PCA → SVM → Result
```

1. Every image is converted to grayscale and resized to 64×64 pixels
2. HOG (Histogram of Oriented Gradients) extracts 7,200 facial features per image
3. PCA reduces those 7,200 features down to 150 key components
4. A Support Vector Machine classifies the face and returns the top 3 matches with confidence scores

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 47.95% |
| Cross-Validation Score | 83.67% |
| Test Images | 853 |
| Celebrities | 20 |
| Training Samples (after augmentation) | 42,081 |

---

## Celebrities in the System

Alexandra Daddario · Robert Downey Jr · Leonardo DiCaprio · Emma Watson  
Scarlett Johansson · Margot Robbie · Tom Holland · Zendaya  
Elon Musk · Gal Gadot · and 10 more

---

## How to Run It Yourself

### 1. Clone the repo
```bash
git clone https://github.com/sadia243/face-recognition-system
cd face-recognition-system
```

### 2. Set up environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get the dataset
Download from Kaggle:  
https://www.kaggle.com/datasets/hereisburak/pins-face-recognition

Unzip it so your folder looks like this:
```
face-recognition-system/
└── 105_classes_pins_dataset/
    ├── pins_Robert Downey Jr/
    ├── pins_Emma Watson/
    └── ...
```

### 4. Train the model
```bash
python train.py
```
This takes around 15–30 minutes. It will save the trained model to `model/trained_model.pkl` automatically.

### 5. Launch the app
```bash
python main.py
```

Upload any photo of a celebrity from the list above and see who it matches.

---

## File Structure

| File | What it does |
|------|-------------|
| `train.py` | Loads dataset, augments data, trains PCA + SVM model |
| `recognise.py` | Loads saved model and identifies a person from a photo |
| `evaluate.py` | Generates accuracy charts and classification report |
| `logger.py` | Logs every recognition result to a CSV file |
| `main.py` | Tkinter GUI — the interface you interact with |
| `requirements.txt` | All Python dependencies |

---

## Tech Stack

- **Python 3.11**
- **OpenCV** — image loading and preprocessing
- **Scikit-learn** — PCA, SVM, GridSearchCV, metrics
- **Scikit-image** — HOG feature extraction
- **Tkinter** — graphical user interface
- **Pandas** — result logging

---

## References

Géron, A., 2019. *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow.* 2nd ed. O'Reilly Media.

Dalal, N. and Triggs, B., 2005. Histograms of oriented gradients for human detection. *CVPR.*
