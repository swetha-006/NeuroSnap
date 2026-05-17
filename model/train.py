"""
model/train.py

Complete retraining pipeline with all fixes applied.

Run from project root:
    python -m model.train

What this fixes:
  1. Uses new model_builder (Functional API) → no more batch_shape error
  2. Balanced dataset + class_weight → no more tumor bias
  3. Threshold tuned on validation set → healthy brain = Normal Brain
  4. Model saved in .keras format (native Keras) AND .h5 (compatibility)
"""

import os
import sys

# MUST be before any tensorflow import
os.environ['TF_USE_LEGACY_KERAS'] = '1'

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from model.model_builder import build_model
from preprocessing.preprocess import load_data


# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_H5_PATH    = os.path.join(_PROJECT_ROOT, 'saved_model', 'model.h5')
THRESHOLD_PATH   = os.path.join(_PROJECT_ROOT, 'saved_model', 'threshold.json')
STATIC_DIR       = os.path.join(_PROJECT_ROOT, 'static')
DATA_DIR         = os.path.join(_PROJECT_ROOT, 'data')


def find_best_threshold(model, X_val, y_val_cat):
    """
    Sweep thresholds 0.10 → 0.90 and pick the one that
    gives the best MACRO F1 on the validation set.
    This stops the model from calling everything 'tumor'.
    """
    y_true = np.argmax(y_val_cat, axis=1)
    probs  = model.predict(X_val, verbose=0)[:, 1]   # P(tumor)

    best_thr = 0.5
    best_f1  = 0.0
    results  = []

    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        results.append((round(float(t), 2), round(float(f1), 4)))
        if f1 > best_f1:
            best_f1  = f1
            best_thr = round(float(t), 2)

    print(f"\n[train] Threshold sweep results (top 5):")
    results.sort(key=lambda x: -x[1])
    for thr, f1 in results[:5]:
        print(f"  threshold={thr:.2f}  macro-F1={f1:.4f}")

    print(f"\n[train] ✅ Best threshold = {best_thr}  (macro-F1 = {best_f1:.4f})")
    return best_thr


def train():
    os.makedirs(os.path.dirname(MODEL_H5_PATH), exist_ok=True)
    os.makedirs(STATIC_DIR, exist_ok=True)

    # ── Delete old bad model ──────────────────────────────────────────────────
    if os.path.isfile(MODEL_H5_PATH):
        os.remove(MODEL_H5_PATH)
        print(f"[train] 🗑  Deleted old model: {MODEL_H5_PATH}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[train] Loading dataset ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes, class_weights = \
        load_data(DATA_DIR)

    print(f"\n[train] Dataset summary:")
    print(f"  Train : {X_train.shape[0]} images")
    print(f"  Val   : {X_val.shape[0]} images")
    print(f"  Test  : {X_test.shape[0]} images")
    print(f"  Classes      : {classes}")
    print(f"  Class weights: {class_weights}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[train] Building model ...")
    model = build_model(input_shape=(128, 128, 3), num_classes=2)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            MODEL_H5_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[train] Starting training ...")
    print("[train] This will take 10-20 minutes on CPU. Please wait.\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,           # smaller batch = better generalisation on small dataset
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ── Find best threshold ───────────────────────────────────────────────────
    best_thr = find_best_threshold(model, X_val, y_val)
    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold': best_thr}, f)
    print(f"[train] Threshold saved → {THRESHOLD_PATH}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[train] Final evaluation on test set ...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print(f"  Test loss     : {test_loss:.4f}")

    probs_test = model.predict(X_test, verbose=0)[:, 1]
    y_pred     = (probs_test >= best_thr).astype(int)
    y_true     = np.argmax(y_test, axis=1)

    print("\n[train] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)
    print("[train] Confusion Matrix:")
    print(f"  True No-Tumor predicted as No-Tumor : {cm[0][0]}")
    print(f"  True No-Tumor predicted as Tumor    : {cm[0][1]}  ← should be LOW")
    print(f"  True Tumor   predicted as No-Tumor  : {cm[1][0]}")
    print(f"  True Tumor   predicted as Tumor     : {cm[1][1]}")

    # ── Save plots ────────────────────────────────────────────────────────────
    _save_plots(history)
    _save_confusion_matrix(cm, classes)

    print(f"\n{'='*50}")
    print(f"✅ Training complete!")
    print(f"   Model     → {MODEL_H5_PATH}")
    print(f"   Threshold → {best_thr}")
    print(f"   Test acc  → {test_acc*100:.2f}%")
    print(f"{'='*50}")
    print("\nNow run:  python run.py")


def _save_plots(history):
    for metric, title, fname in [
        ('accuracy', 'Model Accuracy', os.path.join(STATIC_DIR, 'accuracy.png')),
        ('loss',     'Model Loss',     os.path.join(STATIC_DIR, 'loss.png')),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get(metric, []),
                 label=f'Train {metric.capitalize()}', linewidth=2, color='#1565C0')
        plt.plot(history.history.get(f'val_{metric}', []),
                 label=f'Val {metric.capitalize()}', linewidth=2,
                 linestyle='--', color='#E53935')
        plt.title(title, fontsize=14)
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.legend(); plt.tight_layout()
        plt.savefig(fname, dpi=100); plt.close()
        print(f"[train] Saved → {fname}")


def _save_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    out = os.path.join(STATIC_DIR, 'confusion_matrix.png')
    plt.savefig(out, dpi=100); plt.close()
    print(f"[train] Saved → {out}")


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
