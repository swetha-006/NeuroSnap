# 🧠 Brain Tumor Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

A deep learning-based medical image analysis system that detects brain tumors from MRI scan images using Convolutional Neural Networks (CNN). The system predicts whether a brain MRI scan shows a **tumor** or is **normal**, with a confidence score — all through a simple web interface.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 About the Project

Brain tumors are one of the most critical and life-threatening diseases. Early and accurate detection is essential for effective treatment. Manual analysis of MRI scans by radiologists is time-consuming and prone to human error.

This project automates the detection process using a **Convolutional Neural Network (CNN)** trained on real MRI brain scan images. The trained model is deployed through a **Flask web application**, allowing anyone to upload an MRI image and receive an instant prediction.

### Key Highlights

- Binary classification: **Tumor Detected** vs **Normal Brain**
- Confidence score displayed with every prediction
- MRI image preview in the web app
- Trained model saved as `.h5` for reuse
- Clean, beginner-friendly codebase with comments throughout

---

## 🎬 Demo

```
1. Upload an MRI scan image via the web interface
2. The model processes and analyzes the image
3. Result is displayed: "Tumor Detected" or "Normal Brain"
4. Confidence percentage is shown alongside the result
```

> Web app runs locally at: `http://localhost:5000`

---

## 📁 Project Structure

```
brain_tumor_detection/
│
├── app/                        # Flask web application
│   ├── __init__.py             # App factory
│   ├── routes.py               # URL routes and prediction logic
│   └── templates/
│       └── index.html          # Frontend upload page
│
├── model/                      # ML model scripts
│   ├── model_builder.py        # CNN architecture definition
│   ├── train.py                # Model training script
│   └── predict.py              # Single image prediction
│
├── preprocessing/              # Data preparation
│   └── preprocess.py           # Load, resize, normalize, augment, split
│
├── data/                       # Dataset (not pushed to GitHub)
│   ├── tumor/                  # MRI images with tumor (155 images)
│   └── no_tumor/               # Normal MRI images (98 images)
│
├── saved_model/                # Trained model output
│   └── model.h5                # Saved after training (not pushed to GitHub)
│
├── static/                     # Static assets
│   ├── uploads/                # Temporarily stores uploaded images
│   ├── accuracy_plot.png       # Training accuracy graph
│   └── loss_plot.png           # Training loss graph
│
├── notebooks/
│   └── exploration.ipynb       # EDA and experiments
│
├── requirements.txt            # Python dependencies
├── run.py                      # App entry point
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

```
MRI Image Input
      ↓
Preprocessing (resize 128×128, normalize, augment)
      ↓
CNN Model (Conv2D → MaxPool → Dropout → Dense → Softmax)
      ↓
Prediction Output
      ↓
"Tumor Detected" (with confidence %) OR "Normal Brain"
```

### CNN Concept

The CNN extracts spatial features from MRI images in three stages:

1. **Feature Extraction** — Conv2D layers detect edges, textures, and abnormal tissue patterns
2. **Downsampling** — MaxPooling layers reduce spatial dimensions, retaining key features
3. **Classification** — Dense layers use extracted features to classify tumor vs normal

The final layer uses **Softmax** activation:

```
y = softmax(Wx + b)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| TensorFlow 2.x | Deep learning framework |
| Keras | High-level model building API |
| OpenCV | Image loading and preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Plotting accuracy/loss graphs |
| Flask | Web application framework |
| scikit-learn | Train/test split, metrics |
| Pillow | Image handling in Flask |

---

## 📊 Dataset

**Source:** [Kaggle — Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

| Class | Folder | Images |
|---|---|---|
| Tumor | `data/tumor/` | 155 |
| No Tumor | `data/no_tumor/` | 98 |
| **Total** | | **253** |

### Dataset Setup

1. Download from Kaggle (free account required)
2. Extract the zip file
3. Copy images from `yes/` → `data/tumor/`
4. Copy images from `no/` → `data/no_tumor/`

> **Note:** The `data/` folder is excluded from this repository due to size. You must download and set it up manually as described above.

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up the dataset

Download the dataset from Kaggle and place images in:
```
data/tumor/       ← tumor MRI images
data/no_tumor/    ← normal MRI images
```

### Step 5 — Train the model

```bash
python model/train.py
```

This will:
- Preprocess and split the dataset
- Train the CNN model
- Save the model to `saved_model/model.h5`
- Save accuracy and loss plots to `static/`

Training takes approximately **5–15 minutes** depending on your machine.

### Step 6 — Run the web app

```bash
python run.py
```

Open your browser and go to: `http://localhost:5000`

---

## 💻 Usage

1. Open `http://localhost:5000` in your browser
2. Click **Choose File** and upload an MRI brain scan image (`.jpg`, `.jpeg`, `.png`)
3. Click **Predict**
4. The result is displayed:
   - ✅ **Normal Brain** — No tumor detected
   - ⚠️ **Tumor Detected** — Tumor present, with confidence score

### Predict from command line

```bash
python model/predict.py --image path/to/mri_scan.jpg
```

---

## 🧠 Model Architecture

```
Input Layer        →  (128, 128, 3)
Conv2D (32)        →  ReLU activation
MaxPooling2D       →  (2×2)
Conv2D (64)        →  ReLU activation
MaxPooling2D       →  (2×2)
Conv2D (128)       →  ReLU activation
MaxPooling2D       →  (2×2)
Flatten
Dense (256)        →  ReLU activation
Dropout (0.5)      →  Prevents overfitting
Dense (2)          →  Softmax → [tumor, no_tumor]
```

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Metrics:** Accuracy  
**Epochs:** 25  
**Batch Size:** 32  

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~95% |
| Validation Accuracy | ~88–92% |
| Test Accuracy | ~88–90% |

> Actual results may vary depending on dataset split and training run.

Training graphs (accuracy and loss curves) are saved to `static/` after training.

---

## 🔮 Future Enhancements

- [ ] Multi-class classification (Glioma, Meningioma, Pituitary, No Tumor)
- [ ] Tumor localization with bounding boxes (object detection)
- [ ] Tumor size estimation
- [ ] 3D MRI volume analysis
- [ ] Grad-CAM visualization (heatmap showing where the model looks)
- [ ] REST API for hospital system integration
- [ ] Mobile app (React Native or Flutter)
- [ ] Deploy to cloud (Heroku / AWS / Render)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code is clean and well-commented.

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. It is not intended for clinical or medical use. Always consult a qualified medical professional for diagnosis and treatment.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built by Swetha - CyberSecurity student as a Deep Learning project.

If you found this helpful, give it a ⭐ on GitHub!

---

*"Artificial intelligence in healthcare is not about replacing doctors — it's about empowering them."*
