"""
app/__init__.py

CRITICAL FIX:
  TF_USE_LEGACY_KERAS must be set BEFORE tensorflow is imported anywhere.
  We set it here at the very top of __init__.py so it fires before
  routes.py imports model.predict which imports tensorflow.

  Previously it was set in run.py but Python had already imported
  tensorflow via the 'from app import app' line — too late.
"""

import os

# ── MUST be before ANY tensorflow import in this process ─────────────────────
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# ─────────────────────────────────────────────────────────────────────────────

import secrets
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, g


def create_app():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    app = Flask(
        __name__,
        template_folder='templates',
        static_folder=os.path.join(root, 'static'),
        static_url_path='/static'
    )

    app.secret_key = os.environ.get('SECRET_KEY', 'neurosnap-dev-secret-2024')
    app.config['UPLOAD_FOLDER'] = os.path.join(root, 'static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    from .routes import bp
    app.register_blueprint(bp)

    # File logging
    logs_dir = os.path.join(root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'neurosnap.log')
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    app.logger.addHandler(handler)
    logging.getLogger().addHandler(handler)

    @app.before_request
    def log_request_info():
        import flask
        app.logger.info("Request: %s %s from %s",
                        flask.request.method,
                        flask.request.path,
                        flask.request.remote_addr)
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.context_processor
    def inject_nonce():
        return {'csp_nonce': getattr(g, 'csp_nonce', '')}

    @app.after_request
    def add_csp(response):
        nonce = getattr(g, 'csp_nonce', '')
        nonce_src = f" 'nonce-{nonce}'" if nonce else ''
        csp = (
            "default-src 'self'; "
            f"script-src 'self' https://cdn.jsdelivr.net{nonce_src}; "
            f"style-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com{nonce_src}; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self';"
        )
        response.headers['Content-Security-Policy'] = csp
        return response

    return app


app = create_app()
