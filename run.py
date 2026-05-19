"""
run.py — NeuroSnap entry point

Works for both local development and production (Render/Railway/etc.)
Reads PORT from environment variable set by the hosting platform.
"""

import os
import sys

# MUST be before any tensorflow import
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    print("🧠 NeuroSnap starting …")
    print(f"   Open http://localhost:{port} in your browser")
    print(f"   Debug mode: {debug}")

    app.run(host='0.0.0.0', port=port, debug=debug)
