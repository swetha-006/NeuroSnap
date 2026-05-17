"""
model/predict.py — clean version for retrained model
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any, List, Optional, Tuple

os.environ['TF_USE_LEGACY_KERAS'] = '1'

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJECT_ROOT)

MODEL_PATH     = os.path.join(_PROJECT_ROOT, 'saved_model', 'model.h5')
THRESHOLD_PATH = os.path.join(_PROJECT_ROOT, 'saved_model', 'threshold.json')
CLASS_NAMES: List[str] = ['no_tumor', 'tumor']

_cached_model:     Optional[Any]   = None
_cached_threshold: Optional[float] = None
logger = logging.getLogger(__name__)


class ModelCorruptError(RuntimeError):
    """Raised when model.h5 cannot be loaded."""


def load_trained_model(model_path: str = MODEL_PATH) -> Any:
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: '{model_path}'\n"
            "Train first:  python -m model.train"
        )

    loaded: Optional[Any] = None

    # Strategy 1: tensorflow.keras (should work with new model)
    try:
        from tensorflow.keras.models import load_model as tf_load
        loaded = tf_load(model_path)
        print(f"[predict] ✅ Loaded via tensorflow.keras | output: {loaded.output_shape}")
    except Exception as e1:
        print(f"[predict] tensorflow.keras failed: {e1}")

    # Strategy 2: tf_keras
    if loaded is None:
        try:
            import tf_keras
            loaded = tf_keras.models.load_model(model_path)
            print(f"[predict] ✅ Loaded via tf_keras | output: {loaded.output_shape}")
        except Exception as e2:
            print(f"[predict] tf_keras failed: {e2}")

    # Strategy 3: rebuild + load weights
    if loaded is None:
        try:
            print("[predict] Trying rebuild + load_weights ...")
            from model.model_builder import build_model
            import tensorflow as tf
            fresh = build_model(input_shape=(128, 128, 3), num_classes=2)
            fresh.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            fresh.load_weights(model_path, by_name=True, skip_mismatch=True)
            loaded = fresh
            print(f"[predict] ✅ Loaded via rebuild+weights | output: {loaded.output_shape}")
        except Exception as e3:
            raise ModelCorruptError(
                f"Cannot load model from '{model_path}'.\n"
                f"Error: {e3}\n\n"
                "Delete saved_model/model.h5 and retrain:\n"
                "  python -m model.train"
            ) from e3

    _cached_model = loaded
    return _cached_model


def load_threshold(threshold_path: str = THRESHOLD_PATH) -> float:
    global _cached_threshold
    if _cached_threshold is not None:
        return _cached_threshold

    SAFE_DEFAULT = 0.45

    if os.path.isfile(threshold_path):
        try:
            with open(threshold_path, 'r') as f:
                data = json.load(f)
            val = float(data.get('threshold', SAFE_DEFAULT))
            _cached_threshold = val
        except Exception:
            _cached_threshold = SAFE_DEFAULT
    else:
        _cached_threshold = SAFE_DEFAULT

    print(f"[predict] Decision threshold = {_cached_threshold}")
    return _cached_threshold


def preprocess_image(file_path: str, target_size: Tuple[int, int] = (128, 128)):
    import cv2
    import numpy as np

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image not found: {file_path}")
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Cannot decode: '{file_path}'")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return __import__('numpy').expand_dims(img, axis=0)


def predict_image(
    file_path:   str,
    model:       Optional[Any]       = None,
    class_names: Optional[List[str]] = None,
    threshold:   Optional[float]     = None,
) -> Tuple[str, float, float]:
    import numpy as np

    if class_names is None:
        class_names = CLASS_NAMES

    loaded_model: Any   = load_trained_model() if model is None else model
    thr:          float = load_threshold()     if threshold is None else threshold

    x     = preprocess_image(file_path)
    preds = loaded_model.predict(x, verbose=0)

    print(f"[predict] Raw output: {preds}")

    output = preds[0]
    if hasattr(output, '__len__') and len(output) >= 2:
        tumor_prob = float(output[1])
    elif hasattr(output, '__len__') and len(output) == 1:
        tumor_prob = float(output[0])
    else:
        tumor_prob = float(output)

    tumor_prob = max(0.0, min(1.0, tumor_prob))
    is_tumor   = tumor_prob >= thr
    confidence = round((tumor_prob if is_tumor else 1.0 - tumor_prob) * 100.0, 2)
    tumor_pct  = round(tumor_prob * 100.0, 2)
    label_text = 'Tumor Detected' if is_tumor else 'Normal Brain'

    print(
        f"[predict] {os.path.basename(file_path)} → {label_text}"
        f" | P(tumor)={tumor_pct}% | thr={thr} | conf={confidence}%"
    )
    return label_text, confidence, tumor_pct


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--threshold', type=float, default=None)
    args = p.parse_args()
    try:
        label, conf, tp = predict_image(args.image, threshold=args.threshold)
        print(f"\nResult     : {label}\nConfidence : {conf}%\nP(tumor)   : {tp}%")
    except Exception as e:
        print(f"\n[ERROR] {e}"); sys.exit(1)
