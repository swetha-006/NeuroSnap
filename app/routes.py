"""
app/routes.py — NeuroSnap Flask routes
"""
import os
import uuid
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from werkzeug.utils import secure_filename
from model.predict import predict_image, ModelCorruptError
from PIL import Image

bp = Blueprint('main', __name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/')
def index():
    history = session.get('scan_history', [])
    return render_template('index.html', scan_history=history[-5:][::-1])

@bp.route('/dashboard')
def dashboard():
    history = session.get('scan_history', [])
    return render_template('dashboard.html', scan_history=history)

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file selected.', 'warning')
        return redirect(url_for('main.index'))

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Please upload a valid JPG, PNG or BMP image.', 'warning')
        return redirect(url_for('main.index'))

    filename = secure_filename(file.filename)
    unique_prefix = datetime.utcnow().strftime('%Y%m%d%H%M%S') + '-' + uuid.uuid4().hex[:8]
    filename = f"{unique_prefix}-{filename}"
    save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
    except Exception as e:
        flash(f'Upload failed: {e}', 'danger')
        return redirect(url_for('main.index'))

    # Quick sanity-check: ensure Pillow can open the file (catch corrupted/unsupported files)
    try:
        with Image.open(save_path) as _img:
            _img.verify()
    except Exception as e:
        current_app.logger.exception('Image verification failed for %s', save_path)
        try:
            os.remove(save_path)
        except Exception:
            pass
        flash('Uploaded file is not a valid image (corrupted or unsupported). Please upload a JPG, PNG or BMP.', 'danger')
        return redirect(url_for('main.index'))

    try:
        current_app.logger.info('Running prediction for %s', save_path)
        label, confidence, tumor_pct = predict_image(save_path)
    except FileNotFoundError as e:
        current_app.logger.exception('Model/file not found during prediction for %s', save_path)
        flash(str(e), 'danger')
        return redirect(url_for('main.index'))
    except ModelCorruptError as e:
        current_app.logger.exception('Corrupted model detected while predicting %s', save_path)
        flash(str(e), 'danger')
        return redirect(url_for('main.index'))
    except (ValueError, OSError) as e:
        current_app.logger.exception('Decoding/prediction error for %s', save_path)
        flash(f'Prediction failed ({e.__class__.__name__}): {e}. Try uploading a different JPG/PNG/BMP image.', 'danger')
        return redirect(url_for('main.index'))
    except Exception as e:
        current_app.logger.exception('Unexpected prediction error for %s', save_path)
        flash(f'Prediction error ({e.__class__.__name__}): {e}', 'danger')
        return redirect(url_for('main.index'))

    # Persist to session history (keep last 20)
    history = session.get('scan_history', [])
    history.append({
        'filename':   filename,
        'result':     label,
        'confidence': confidence,
        'tumor_pct':  tumor_pct,
        'timestamp':  datetime.now().strftime('%d %b %H:%M'),
    })
    session['scan_history'] = history[-20:]

    return render_template('result.html',
                           filename=filename,
                           result=label,
                           confidence=confidence,
                           tumor_pct=tumor_pct)
