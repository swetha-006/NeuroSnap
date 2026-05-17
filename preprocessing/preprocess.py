"""
preprocessing/preprocess.py

Load MRI images, preprocess, fix class imbalance via oversampling,
and split into train / validation / test sets.

ROOT CAUSE FIX:
  The dataset has 155 tumor vs 98 no_tumor images (61% vs 39%).
  Without fixing this, the model learns to predict "tumor" for everything
  because that gives a higher raw accuracy.  We fix it three ways:
    1. Class-weight calculation  → passed to model.fit()
    2. Oversampling no_tumor     → balance raw counts before augmentation
    3. Stratified splits         → keep ratio consistent in every split

Supports two dataset folder layouts automatically:
  Layout A (flat):    data/tumor/  and  data/no_tumor/
  Layout B (nested):  data/brain_tumor_dataset/yes/  and  data/brain_tumor_dataset/no/
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Absolute project root so default data_dir works regardless of CWD
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_DEFAULT_DATA = os.path.join(_PROJECT_ROOT, 'data')


# ── Folder layout detection ──────────────────────────────────────────────────

def _resolve_folders(data_dir):
    """Return (tumor_folder, no_tumor_folder) detecting layout automatically."""
    layout_a_t  = os.path.join(data_dir, 'tumor')
    layout_a_nt = os.path.join(data_dir, 'no_tumor')
    layout_b_t  = os.path.join(data_dir, 'brain_tumor_dataset', 'yes')
    layout_b_nt = os.path.join(data_dir, 'brain_tumor_dataset', 'no')

    if os.path.isdir(layout_a_t) and os.path.isdir(layout_a_nt):
        print(f"[preprocess] Layout A detected → {layout_a_t}  |  {layout_a_nt}")
        return layout_a_t, layout_a_nt
    if os.path.isdir(layout_b_t) and os.path.isdir(layout_b_nt):
        print(f"[preprocess] Layout B detected → {layout_b_t}  |  {layout_b_nt}")
        return layout_b_t, layout_b_nt

    raise FileNotFoundError(
        f"Dataset not found under '{data_dir}'.\n"
        "Expected:\n"
        "  data/tumor/  +  data/no_tumor/\n"
        "  OR\n"
        "  data/brain_tumor_dataset/yes/  +  data/brain_tumor_dataset/no/"
    )


# ── Image loader ─────────────────────────────────────────────────────────────

def _load_images_from_folder(folder, label, img_size):
    """Load all valid images from a folder, return (images, labels)."""
    VALID_EXT = ('.png', '.jpg', '.jpeg', '.bmp')
    images, labels = [], []
    skipped = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith(VALID_EXT):
            continue
        fpath = os.path.join(folder, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                skipped += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label)
        except Exception:
            skipped += 1
    if skipped:
        print(f"  [warn] Skipped {skipped} unreadable files in {folder}")
    return images, labels


# ── Main load_data ────────────────────────────────────────────────────────────

def load_data(
    data_dir=None,
    img_size=(128, 128),
    val_size=0.15,
    test_size=0.15,
    augment=True,
    augment_factor=2,        # ← increased from 1 to help no_tumor class
    balance=True,            # ← NEW: oversample minority class
):
    """
    Load, preprocess, balance, augment and split MRI images.

    Args:
        data_dir       : dataset root directory.
        img_size       : (width, height) for resizing.
        val_size       : fraction of total data for validation.
        test_size      : fraction of total data for test.
        augment        : apply augmentation to training set.
        augment_factor : augmented variants per training image.
        balance        : oversample minority class to fix class imbalance.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, class_weights
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA
    tumor_folder, no_tumor_folder = _resolve_folders(data_dir)

    # ── Load images ──────────────────────────────────────────────────────────
    # IMPORTANT: class 0 = no_tumor, class 1 = tumor
    # This order MUST match what predict.py uses when interpreting argmax output
    no_tumor_imgs, no_tumor_lbls = _load_images_from_folder(no_tumor_folder, label=0, img_size=img_size)
    tumor_imgs,    tumor_lbls    = _load_images_from_folder(tumor_folder,    label=1, img_size=img_size)

    print(f"[preprocess] Loaded → no_tumor={len(no_tumor_imgs)}, tumor={len(tumor_imgs)}")

    # ── Fix class imbalance: oversample minority class ───────────────────────
    # ROOT CAUSE FIX #1: without this the model biases toward predicting tumor
    if balance and len(no_tumor_imgs) < len(tumor_imgs):
        deficit   = len(tumor_imgs) - len(no_tumor_imgs)
        extra_idx = np.random.choice(len(no_tumor_imgs), size=deficit, replace=True)
        no_tumor_imgs = no_tumor_imgs + [no_tumor_imgs[i] for i in extra_idx]
        no_tumor_lbls = no_tumor_lbls + [0] * deficit
        print(f"[preprocess] After oversampling → no_tumor={len(no_tumor_imgs)}, tumor={len(tumor_imgs)}")

    # ── Combine and shuffle ──────────────────────────────────────────────────
    X = np.array(no_tumor_imgs + tumor_imgs, dtype='float32')
    y = np.array(no_tumor_lbls + tumor_lbls, dtype='int32')

    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    print(f"[preprocess] Total images: {len(X)}  "
          f"(no_tumor={int((y==0).sum())}, tumor={int((y==1).sum())})")

    # ── Stratified splits ────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp, random_state=42
    )
    print(f"[preprocess] Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ── Class weights (ROOT CAUSE FIX #2) ────────────────────────────────────
    # Passed to model.fit() so the loss function penalises mistakes on the
    # minority class more heavily even after oversampling.
    unique_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    print(f"[preprocess] Class weights: {class_weights}")

    # ── Augmentation ─────────────────────────────────────────────────────────
    if augment and augment_factor > 0:
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest'
        )
        aug_imgs, aug_lbls = [], []
        for i in range(len(X_train)):
            x_i = X_train[i:i+1]
            gen = datagen.flow(x_i, batch_size=1)
            for _ in range(augment_factor):
                aug_imgs.append(next(gen)[0])
                aug_lbls.append(y_train[i])
        if aug_imgs:
            X_train = np.concatenate([X_train, np.array(aug_imgs)], axis=0)
            y_train = np.concatenate([y_train, np.array(aug_lbls)], axis=0)
        print(f"[preprocess] After augmentation → train={len(X_train)}")

    # ── One-hot encode ───────────────────────────────────────────────────────
    num_classes = 2
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes=num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=num_classes)

    class_names = ['no_tumor', 'tumor']
    print(f"\n✅ Preprocessing complete. Classes: {class_names}")
    return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat), class_names, class_weights


if __name__ == '__main__':
    try:
        (Xtr, ytr), (Xv, yv), (Xte, yte), classes, cw = load_data()
        print(f"X_train={Xtr.shape}, X_val={Xv.shape}, X_test={Xte.shape}")
        print(f"Class weights: {cw}")
    except Exception as e:
        print(f"[ERROR] {e}")
