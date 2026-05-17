<div align="center">

<img src="https://img.shields.io/badge/🧠-NeuroSnap-1a237e?style=for-the-badge&labelColor=283593" alt="NeuroSnap"/>

# NeuroSnap — Brain Tumor Detection

**AI-powered MRI analysis using Convolutional Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=flat-square)]()

[Live Demo](#deployment) · [Quick Start](#quick-start) · [Dataset](#dataset) · [Model](#model-architecture) · [Deployment](#deployment)

---

> ⚠️ **Medical Disclaimer:** NeuroSnap is built for **educational and research purposes only**.  
> It is **not** a certified medical device and must **never** be used as a substitute for professional medical diagnosis.

</div>

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the App](#running-the-app)
- [Deployment](#deployment)
- [API Routes](#api-routes)
- [Technologies Used](#technologies-used)
- [Known Issues & Fixes](#known-issues--fixes)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 About the Project

**NeuroSnap** is a full-stack deep learning web application that detects brain tumors from MRI scan images using a custom Convolutional Neural Network (CNN). The system:

- Accepts MRI scan images via a web interface
- Preprocesses and analyses them through a trained CNN
- Returns a clear prediction — **Tumor Detected** or **Normal Brain**
- Shows a **confidence score** and **raw tumor probability**
- Maintains a **session-based scan history** and dashboard

Built as a complete end-to-end ML engineering project covering data preprocessing, model training, evaluation, and web deployment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 MRI Analysis | Upload any JPG/PNG/BMP MRI scan and get instant prediction |
| 📊 Confidence Score | See exactly how confident the model is |
| 📈 Dashboard | Track all scans in the current session with stats |
| 🔁 Scan History | Last 20 scans stored in session |
| 🎨 Professional UI | Responsive design built with Bootstrap 5 |
| ⚡ Fast Inference | Model cached in memory — no reload on every request |
| 🛡️ Input Validation | Pillow-based image verification before inference |
| 📝 Logging | Rotating file logs for all requests and errors |

---

## 📁 Project Structure

```
NeuroSnap/
│
├── app/                            # Flask web application
│   ├── __init__.py                 # App factory + CSP headers + logging
│   ├── routes.py                   # URL routes: /, /predict, /dashboard, /about
│   └── templates/
│       ├── base.html               # Shared layout with navbar and footer
│       ├── index.html              # MRI upload page with drag-and-drop
│       ├── result.html             # Prediction result with confidence gauge
│       ├── dashboard.html          # Scan stats, charts, history table
│       └── about.html             # Project info and CNN architecture visual
│
├── model/                          # Deep learning model
│   ├── __init__.py
│   ├── model_builder.py            # CNN architecture (Functional API)
│   ├── train.py                    # Training pipeline with class balancing
│   └── predict.py                  # Inference with 3-strategy model loading
│
├── preprocessing/                  # Data pipeline
│   ├── __init__.py
│   └── preprocess.py               # Load, balance, augment, split dataset
│
├── data/                           # Dataset (not committed to Git)
│   ├── tumor/                      # MRI images WITH tumor (155 images)
│   └── no_tumor/                   # Normal MRI images (98 images)
│
├── saved_model/                    # Trained model output (not committed to Git)
│   ├── model.h5                    # Best model weights (saved during training)
│   └── threshold.json              # Tuned decision threshold (saved during training)
│
├── static/
│   ├── neurosnap.css               # Custom styles
│   ├── neurosnap.js                # Upload preview, spinner, flash auto-dismiss
│   ├── uploads/                    # Temporarily stores uploaded MRI images
│   ├── accuracy.png                # Training accuracy plot (generated)
│   ├── loss.png                    # Training loss plot (generated)
│   └── confusion_matrix.png        # Confusion matrix (generated)
│
├── notebooks/
│   └── exploration.ipynb           # EDA and experimentation
│
├── .github/
│   └── workflows/
│       └── python-package-conda.yml # CI workflow
│
├── .gitignore
├── requirements.txt
├── run.py                          # App entry point → python run.py
└── README.md
```

---

## ⚙️ How It Works

```
User uploads MRI image
        ↓
Flask receives file → validates format (Pillow verify)
        ↓
OpenCV loads image → resize 128×128 → normalize [0,1]
        ↓
CNN model runs inference → outputs [P(no_tumor), P(tumor)]
        ↓
P(tumor) compared against tuned threshold
        ↓
     ┌──────────────────────────────┐
     │  P(tumor) ≥ threshold        │   →   ⚠️  Tumor Detected
     │  P(tumor) < threshold        │   →   ✅  Normal Brain
     └──────────────────────────────┘
        ↓
Result + confidence + tumor % shown to user
Scan saved to session history → Dashboard updated
```

### Class Imbalance Fix

The dataset has **155 tumor vs 98 no-tumor** images (61/39 split). Without correction, the model learns to always predict "Tumor" for a free 61% accuracy. NeuroSnap fixes this three ways:

1. **Oversampling** — minority class images duplicated to match majority count
2. **Class weights** — loss function penalises tumor-class mistakes more during training
3. **Threshold tuning** — after training, threshold is swept from 0.10–0.90 and the value maximising macro-F1 on the validation set is saved to `threshold.json`

---

## 🏗️ Model Architecture

Built using **Keras Functional API** (not Sequential) to avoid the `batch_shape` serialisation bug in TF 2.16+.

```
Input (128, 128, 3)
    │
    ├── Conv2D(32, 3×3, relu) → BatchNorm → MaxPool(2×2)
    │
    ├── Conv2D(64, 3×3, relu) → BatchNorm → MaxPool(2×2)
    │
    ├── Conv2D(128, 3×3, relu) → BatchNorm → MaxPool(2×2)
    │
    ├── Conv2D(256, 3×3, relu) → BatchNorm → MaxPool(2×2)
    │
    ├── Flatten
    │
    ├── Dense(256, relu) → Dropout(0.5)
    │
    ├── Dense(128, relu) → Dropout(0.3)
    │
    └── Dense(2, softmax)  →  [P(no_tumor), P(tumor)]
```

| Parameter | Value |
|---|---|
| Input size | 128 × 128 × 3 |
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Epochs | Up to 50 (EarlyStopping) |
| Batch size | 16 |
| Callbacks | ModelCheckpoint, EarlyStopping, ReduceLROnPlateau |

---

## 📊 Dataset

**Source:** [Kaggle — Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

| Class | Folder | Count |
|---|---|---|
| No Tumor | `data/no_tumor/` | 98 images |
| Tumor | `data/tumor/` | 155 images |
| **Total** | | **253 images** |

> The dataset is **not committed to Git** (too large). Follow [Dataset Setup](#dataset-setup) below.

### Dataset Setup

1. Go to [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
2. Click **Download** (free Kaggle account required)
3. Extract the zip file
4. Copy images:
   - `yes/` folder images → `data/tumor/`
   - `no/` folder images → `data/no_tumor/`

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/NeuroSnap.git
cd NeuroSnap

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv          # Windows
python3.11 -m venv venv        # Mac/Linux

# 3. Activate
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Set up dataset (see Dataset section above)

# 6. Train the model
python -m model.train

# 7. Run the app
python run.py
```

Open → **http://localhost:5000**

---

## 🔧 Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | **3.11** (3.12+ NOT supported by TensorFlow) |
| pip | Latest |
| Git | Any |
| RAM | 4GB minimum, 8GB recommended |

### Step-by-step

**1. Clone**
```bash
git clone https://github.com/YOUR_USERNAME/NeuroSnap.git
cd NeuroSnap
```

**2. Python 3.11 virtual environment**
```bash
# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Verify TensorFlow**
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.x.x
```

---

## 🏋️ Training the Model

```bash
python -m model.train
```

Training output:
```
[preprocess] Layout A detected → data/tumor  |  data/no_tumor
[preprocess] After oversampling → no_tumor=155, tumor=155
[preprocess] Split → train=217, val=46, test=47
[preprocess] Class weights: {0: 1.0, 1: 1.0}

Epoch 1/50
14/14 ━━━━━━━━━━━━━━━━━━━━ loss: 0.6821 - accuracy: 0.5819
...
Epoch 28/50
14/14 ━━━━━━━━━━━━━━━━━━━━ loss: 0.1423 - accuracy: 0.9491

Best threshold = 0.42  (macro-F1 = 0.9234)

Test Accuracy : 91.48%

Classification Report:
              precision  recall  f1-score
   no_tumor      0.93    0.89      0.91
      tumor       0.91    0.94      0.92

✅ Training complete!
```

> ⏱ Training takes **10–20 minutes** on CPU depending on your machine.

### After training, these files are generated:
- `saved_model/model.h5` — best model weights
- `saved_model/threshold.json` — optimal decision threshold
- `static/accuracy.png` — accuracy curve
- `static/loss.png` — loss curve
- `static/confusion_matrix.png` — confusion matrix

---

## 🌐 Running the App

```bash
python run.py
```

```
🧠 NeuroSnap starting …
   Open http://localhost:5000 in your browser
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Pages

| Route | Page | Description |
|---|---|---|
| `/` | Home | MRI upload form with scan history |
| `/predict` | Result | Prediction result with confidence |
| `/dashboard` | Dashboard | Stats, charts, full scan history |
| `/about` | About | Project info and CNN diagram |

---

## ☁️ Deployment

### Option 1 — Render (Recommended, Free)

1. Push your code to GitHub (see [Git Setup](#git-setup))

2. Go to [render.com](https://render.com) → **New Web Service**

3. Connect your GitHub repo

4. Configure:
   ```
   Name:          neurosnap
   Runtime:       Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn run:app
   ```

5. Add environment variable:
   ```
   SECRET_KEY = your-random-secret-key-here
   ```

6. Click **Deploy** — live URL in ~3 minutes

> ⚠️ The model.h5 file (~100MB) must be committed or uploaded separately.  
> Render's free tier has 512MB RAM — sufficient for inference.

---

### Option 2 — Railway

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. Deploy:
   ```bash
   railway init
   railway up
   ```

3. Set environment variable:
   ```bash
   railway variables set SECRET_KEY=your-secret-key
   ```

---

### Option 3 — PythonAnywhere (Free tier)

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your project via Files tab or Git clone
3. Create a new Web App → Flask → Python 3.11
4. Set WSGI file to point to `run.py`
5. Install requirements in a virtualenv via Bash console

---

### Option 4 — Local (Production mode)

```bash
pip install gunicorn
gunicorn --workers 2 --bind 0.0.0.0:5000 run:app
```

---

### Pre-deployment checklist

```bash
# Add gunicorn to requirements
echo "gunicorn>=21.0.0" >> requirements.txt

# Set a real secret key
export SECRET_KEY="your-very-long-random-secret-key"

# Test production mode locally
gunicorn run:app
```

---

## 🗂️ Git Setup

### First push

```bash
git init
git add .
git commit -m "Initial commit: NeuroSnap Brain Tumor Detection"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NeuroSnap.git
git push -u origin main
```

### Important — what is NOT pushed to Git

The `.gitignore` excludes these (too large / generated):

```
data/           ← dataset (download from Kaggle separately)
saved_model/    ← model.h5 and threshold.json (generated by training)
venv/           ← virtual environment
__pycache__/    ← Python bytecode
static/uploads/ ← uploaded MRI images
static/*.png    ← training plots (generated)
```

### Pushing the trained model (for deployment)

Since `model.h5` can be ~100MB, use **Git LFS**:

```bash
# Install Git LFS
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add saved_model/model.h5
git add saved_model/threshold.json
git commit -m "Add trained model"
git push
```

Or alternatively, upload `model.h5` manually via your deployment platform's dashboard.

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Core language |
| TensorFlow | 2.13+ | Deep learning framework |
| tf_keras | 2.13+ | Keras 2 API compatibility |
| Keras | 2.x | Model building and training |
| OpenCV | 4.8+ | Image loading and preprocessing |
| NumPy | 1.24+ | Numerical operations |
| scikit-learn | 1.3+ | Metrics, class weights, splitting |
| Matplotlib | 3.7+ | Training plots |
| Flask | 2.3+ | Web framework |
| Werkzeug | 2.3+ | Secure file uploads |
| Pillow | 10.0+ | Image verification |
| Bootstrap | 5.3 | Frontend UI |
| Chart.js | CDN | Dashboard charts |
| Gunicorn | 21.0+ | Production WSGI server |

---

## 🐛 Known Issues & Fixes

### `TypeError: Unrecognized keyword arguments: ['batch_shape']`
**Cause:** Model was saved by TF 2.16+ with incompatible config format.  
**Fix:** Delete `saved_model/model.h5` and retrain:
```bash
python -m model.train
```

### Model predicts Tumor for every image
**Cause:** Class imbalance — dataset has 61% tumor images.  
**Fix:** Already handled in `preprocess.py` (oversampling + class_weight) and `train.py` (threshold tuning). Ensure you retrain with the latest code.

### `tensorflow` not found / install fails
**Cause:** Python 3.12+ is not supported.  
**Fix:** Use Python **3.11** specifically:
```bash
py -3.11 -m venv venv
```

### Port 5000 already in use
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -i :5000 && kill -9 <PID>

# Or change port in run.py
app.run(port=5001)
```

---

## 🔮 Future Enhancements

- [ ] **Multi-class classification** — Glioma, Meningioma, Pituitary, No Tumor
- [ ] **Grad-CAM heatmaps** — visualise where the model is looking on the MRI
- [ ] **Tumor segmentation** — outline the tumor region on the scan
- [ ] **3D MRI analysis** — volumetric scan support
- [ ] **REST API** — JSON endpoint for hospital system integration
- [ ] **Mobile app** — React Native / Flutter frontend
- [ ] **Model versioning** — MLflow or DVC integration
- [ ] **GPU training** — CUDA support for faster training
- [ ] **Transfer learning** — Fine-tune EfficientNet or ResNet50

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

Please ensure:
- Code is clean and well-commented
- No dataset or model files are committed
- `requirements.txt` is updated if new packages are added

---

## 📄 License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**NeuroSnap** — built from scratch as a complete ML engineering project.

> If this helped you, give it a ⭐ on GitHub!

---

<div align="center">

*"The goal of AI in healthcare is not to replace the physician,*  
*but to give the physician superpowers."*

**Built with ❤️ using Python, TensorFlow, and Flask**

</div>
