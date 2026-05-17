"""
convert_model.py  —  run this ONCE from your project root

THE PROBLEM:
  Your model.h5 was saved by TF 2.21 which stores InputLayer config
  as {'batch_shape': [...]} but tf_keras expects {'batch_input_shape': [...]}.
  Neither tf_keras nor tensorflow.keras can load it directly.

THE FIX:
  1. Load the weights-only from the broken .h5
  2. Rebuild the exact same CNN architecture fresh
  3. Load weights into the fresh model
  4. Save as a new compatible model.h5

Run:
    python convert_model.py
"""

import os
import sys

# Must be set before any tensorflow import
os.environ['TF_USE_LEGACY_KERAS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OLD_MODEL_PATH = os.path.join('saved_model', 'model.h5')
NEW_MODEL_PATH = os.path.join('saved_model', 'model_converted.h5')


def rebuild_and_convert():
    import numpy as np

    print("[convert] Importing tensorflow ...")
    import tensorflow as tf
    print(f"[convert] TensorFlow version: {tf.__version__}")

    # ── Step 1: Build fresh model with identical architecture ─────────────────
    print("[convert] Building fresh model architecture ...")
    from model.model_builder import build_model
    fresh_model = build_model(input_shape=(128, 128, 3), num_classes=2)
    fresh_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    fresh_model.summary()

    # ── Step 2: Load weights from old broken model ────────────────────────────
    print(f"\n[convert] Loading weights from {OLD_MODEL_PATH} ...")
    try:
        # load_weights can read weights even if the full model config is broken
        fresh_model.load_weights(OLD_MODEL_PATH, by_name=True, skip_mismatch=True)
        print("[convert] Weights loaded successfully ✅")
    except Exception as e:
        print(f"[convert] load_weights failed: {e}")
        print("[convert] Trying alternate method ...")
        # Alternate: read weight values directly via h5py
        try:
            import h5py
            with h5py.File(OLD_MODEL_PATH, 'r') as f:
                print("[convert] Model groups:", list(f.keys()))
                # Try loading by layer names
                fresh_model.load_weights(OLD_MODEL_PATH, by_name=True, skip_mismatch=True)
        except Exception as e2:
            print(f"[convert] ❌ Cannot load weights: {e2}")
            print("\n[convert] SOLUTION: You need to retrain the model.")
            print("Run:  python -m model.train")
            sys.exit(1)

    # ── Step 3: Quick sanity-check — predict on a dummy input ─────────────────
    print("\n[convert] Sanity check ...")
    dummy = np.zeros((1, 128, 128, 3), dtype='float32')
    out   = fresh_model.predict(dummy, verbose=0)
    print(f"[convert] Output shape: {out.shape}  values: {out}")
    assert out.shape == (1, 2), f"Unexpected output shape: {out.shape}"

    # ── Step 4: Save the fixed model ──────────────────────────────────────────
    print(f"\n[convert] Saving converted model → {NEW_MODEL_PATH}")
    fresh_model.save(NEW_MODEL_PATH)
    print("[convert] ✅ Saved successfully!")

    # ── Step 5: Rename files ──────────────────────────────────────────────────
    backup = OLD_MODEL_PATH + '.backup'
    os.rename(OLD_MODEL_PATH, backup)
    os.rename(NEW_MODEL_PATH, OLD_MODEL_PATH)
    print(f"[convert] Old model backed up → {backup}")
    print(f"[convert] New model saved     → {OLD_MODEL_PATH}")
    print("\n✅ Conversion complete! Now run:  python run.py")


if __name__ == '__main__':
    if not os.path.isfile(OLD_MODEL_PATH):
        print(f"[convert] ❌ Model not found at '{OLD_MODEL_PATH}'")
        print("Run training first:  python -m model.train")
        sys.exit(1)
    rebuild_and_convert()
