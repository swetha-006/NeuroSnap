"""
run.py — NeuroSnap entry point

NOTE: TF_USE_LEGACY_KERAS is now set inside app/__init__.py
which is the earliest safe point before tensorflow loads.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("🧠 NeuroSnap starting …")
    print("   Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
