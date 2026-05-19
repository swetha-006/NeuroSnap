# saved_model/

Place your trained model here:
  - `model.h5`      — Keras HDF5 model (output of model/train.py)
  - `threshold.json` — {"threshold": 0.XX}  (output of model/train.py)

To train from scratch, run from project root:
    python -m model.train

The training script will automatically write both files here.
