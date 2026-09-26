"""K.I.S.A.N. AI - Application Launcher.

Quick-start entrypoint to launch the web dashboard and REST API directly from project root:
    python app.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ui.app import app

if __name__ == "__main__":
    print("[K.I.S.A.N. AI] Launching Web Dashboard on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
